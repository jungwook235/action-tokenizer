"""Convert a SAM3 mask plane ((T,H,W) member) from the legacy npz to per-frame layout.

Why: the training loader needs ONE frame (the x1 step) per sample, but np.load of
the legacy layout inflates the whole episode (~24MB) to slice 64KB. The per-frame
layout stores one packbits member per frame so only that frame is inflated
(~13x faster sample loads on Lustre).

--role picks the source plane: "union" (default, unchanged — member 'mask'),
"robot" ('robot_mask') or "object" ('object_mask'); the role-split planes exist only
in npz written by the role-split batch runs (batch_sam3_gr1_unified.py). A non-union
role defaults its output to "masks_pf_<role>" and stamps a 'role' member, which the
training loader cross-checks against --mask-role. Union output is unchanged (no 'role'
member) so existing mirrors and their skip logic behave exactly as before.

Output (spec-fixed):
  <root>/<dataset_dir>/<dst_subdir>/chunk-XXX/<video_key>/episode_XXXXXX.npz
    - member 'f{t}' (t = 0..T-1): np.packbits(mask[t].reshape(-1)) — 1-D uint8
    - member 'meta': np.array([T, H, W], dtype=np.int32)
    - member 'role': the plane name, ONLY when --role != union
  Recover: np.unpackbits(z[f"f{t}"])[:H*W].reshape(H, W)

The source subdir is NEVER touched (read-only); writes are atomic
(<out>.tmp → os.replace), and existing outputs whose meta T matches the input are
skipped unless --overwrite.
"""

import argparse
import os
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np


ROLE_KEYS = {"union": "mask", "robot": "robot_mask", "object": "object_mask"}


def convert_one(job):
    src, dst, overwrite, role = job
    key = ROLE_KEYS[role]
    try:
        src_bytes = os.path.getsize(src)
        with np.load(src) as z:
            if key not in z.files:
                return ("fail", f"{src}: no member {key!r} (has {sorted(z.files)})", 0, 0)
            mask = z[key]  # (T, H, W) uint8 {0,1}
        T, H, W = mask.shape

        if os.path.isfile(dst):
            # Never overwrite a mirror built from a DIFFERENT plane: same filenames,
            # different content. Refuse loudly instead of silently replacing it.
            try:
                with np.load(dst) as zd:
                    dst_role = str(zd["role"]) if "role" in zd.files else "union"
                    meta = zd["meta"]
            except Exception:
                dst_role, meta = role, None  # unreadable/partial output → regenerate
            if dst_role != role:
                return ("fail", f"{dst}: holds the {dst_role!r} plane, refusing to "
                                f"overwrite with {role!r}", 0, 0)
            if not overwrite and meta is not None and int(meta[0]) == T:
                return ("skip", src, src_bytes, os.path.getsize(dst))

        members = {f"f{t}": np.packbits(mask[t].reshape(-1)) for t in range(T)}
        members["meta"] = np.array([T, H, W], dtype=np.int32)
        if role != "union":
            members["role"] = np.array(role)

        os.makedirs(os.path.dirname(dst), exist_ok=True)
        # np.savez_compressed APPENDS ".npz" when the name does not end with it,
        # so the tmp name must keep the suffix for os.replace to find the file.
        tmp = dst + ".tmp.npz"
        np.savez_compressed(tmp, **members)
        os.replace(tmp, dst)
        return ("done", src, src_bytes, os.path.getsize(dst))
    except Exception:
        return ("fail", f"{src}: {traceback.format_exc(limit=1).strip()}", 0, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--src-subdir", default="masks")
    ap.add_argument("--dst-subdir", default=None,
                    help="default: masks_pf for --role union, masks_pf_<role> otherwise")
    ap.add_argument("--role", default="union", choices=sorted(ROLE_KEYS),
                    help="which SAM3 plane to convert (default: union = member 'mask')")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="debug: convert at most N episodes")
    args = ap.parse_args()
    if args.dst_subdir is None:
        args.dst_subdir = "masks_pf" if args.role == "union" else f"masks_pf_{args.role}"
    assert args.dst_subdir != args.src_subdir, "--dst-subdir must differ from --src-subdir"

    root = Path(args.root)
    srcs = sorted(root.glob(f"*/{args.src_subdir}/chunk-*/*/episode_*.npz"))
    if args.limit:
        srcs = srcs[: args.limit]
    jobs = []
    for s in srcs:
        rel = s.relative_to(root)
        parts = list(rel.parts)
        assert parts[1] == args.src_subdir, rel
        parts[1] = args.dst_subdir
        jobs.append((str(s), str(root.joinpath(*parts)), args.overwrite, args.role))
    print(f"[convert] {len(jobs)} episodes, workers={args.workers}, role={args.role} "
          f"({ROLE_KEYS[args.role]}), {args.src_subdir} -> {args.dst_subdir}", flush=True)

    t0 = time.time()
    counts = {"done": 0, "skip": 0, "fail": 0}
    in_bytes = out_bytes = 0
    failures = []
    with Pool(args.workers) as pool:
        for i, (status, src, sb, db) in enumerate(pool.imap_unordered(convert_one, jobs, chunksize=8), 1):
            counts[status] += 1
            in_bytes += sb
            out_bytes += db
            if status == "fail":
                failures.append(src)
            if i % 1000 == 0 or i == len(jobs):
                dt = time.time() - t0
                print(f"[convert] {i}/{len(jobs)} done={counts['done']} skip={counts['skip']} "
                      f"fail={counts['fail']} elapsed={dt:.0f}s", flush=True)

    print(f"[convert] FINISHED in {time.time()-t0:.0f}s  "
          f"done={counts['done']} skip={counts['skip']} fail={counts['fail']}")
    print(f"[convert] total input {in_bytes/2**30:.2f} GiB -> output {out_bytes/2**30:.2f} GiB "
          f"(ratio {out_bytes/max(in_bytes,1):.3f}x)")
    if failures:
        print(f"[convert] FAILED episodes ({len(failures)}):")
        for f in failures:
            print("  ", f)
        sys.exit(1)


if __name__ == "__main__":
    main()
