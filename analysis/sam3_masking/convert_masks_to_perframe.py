"""Convert SAM3 union-mask npz (one (T,H,W) 'mask' member) to per-frame layout.

Why: the training loader needs ONE frame (the x1 step) per sample, but np.load of
the legacy layout inflates the whole episode (~24MB) to slice 64KB. The per-frame
layout stores one packbits member per frame so only that frame is inflated
(~13x faster sample loads on Lustre).

Output (spec-fixed):
  <root>/<dataset_dir>/<dst_subdir>/chunk-XXX/<video_key>/episode_XXXXXX.npz
    - member 'f{t}' (t = 0..T-1): np.packbits(mask[t].reshape(-1)) — 1-D uint8
    - member 'meta': np.array([T, H, W], dtype=np.int32)
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


def convert_one(job):
    src, dst, overwrite = job
    try:
        src_bytes = os.path.getsize(src)
        with np.load(src) as z:
            mask = z["mask"]  # (T, H, W) uint8 {0,1}
        T, H, W = mask.shape

        if not overwrite and os.path.isfile(dst):
            try:
                with np.load(dst) as zd:
                    meta = zd["meta"]
                if int(meta[0]) == T:
                    return ("skip", src, src_bytes, os.path.getsize(dst))
            except Exception:
                pass  # unreadable/partial output → regenerate

        members = {f"f{t}": np.packbits(mask[t].reshape(-1)) for t in range(T)}
        members["meta"] = np.array([T, H, W], dtype=np.int32)

        os.makedirs(os.path.dirname(dst), exist_ok=True)
        # np.savez_compressed APPENDS ".npz" when the name does not end with it,
        # so the tmp name must keep the suffix for os.replace to find the file.
        tmp = dst + ".tmp.npz"
        np.savez_compressed(tmp, **members)
        os.replace(tmp, dst)
        return ("done", src, src_bytes, os.path.getsize(dst))
    except Exception:
        return ("fail", src, 0, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--src-subdir", default="masks")
    ap.add_argument("--dst-subdir", default="masks_pf")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="debug: convert at most N episodes")
    args = ap.parse_args()

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
        jobs.append((str(s), str(root.joinpath(*parts)), args.overwrite))
    print(f"[convert] {len(jobs)} episodes, workers={args.workers}, "
          f"{args.src_subdir} -> {args.dst_subdir}", flush=True)

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
