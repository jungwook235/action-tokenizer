"""Batch SAM3 masking over the gr1_unified datasets, with robot/object masks kept apart.

Same prompt combo as the existing
`PhysicalAI-Robotics-GR00T-X-Embodiment-Sim_sam3_D_parts_nouns_norobot` output
(D_parts_nouns_norobot = ["robot hand", "robot arm", *task_nouns.json[task]]), but
the npz now records which mask came from which prompt group. The old npz stores
only the union `mask`, and a union cannot be split after the fact — hence this
re-run rather than a converter.

Outputs mirror the source lerobot layout, one subdir per dataset:

  <out_root>/<dataset_dirname>/
      cutout/chunk-000/observation.images.ego_view/episode_XXXXXX.mp4
      overlay/chunk-000/observation.images.ego_view/episode_XXXXXX.mp4
      masks/chunk-000/observation.images.ego_view/episode_XXXXXX.npz
  <out_root>/manifest_shard{K}.jsonl

npz keys: see sam3_batch_core (mask / robot_mask / object_mask / prompt_masks /
prompts / prompt_roles / n_robot_prompts / episode_index / fps).

`--skip-videos` reuses the cutout/overlay mp4s already produced by the previous
run and writes only the npz — the prompts are identical, so the union mask and
therefore the cutout are unchanged; this cuts the mp4 encoding work.

Sharding: episodes are enumerated in a fixed sorted order across all dataset dirs;
episode j goes to shard (j % num_shards). Existing outputs are skipped, so shards
are resumable and can be re-run with a different shard count.

Example:
  PYTHONPATH=venv_sam3/lib/python3.10/site-packages \
    ~/miniconda3/envs/gr00t/bin/python batch_sam3_gr1_unified.py \
      --out-root /path/to/..._sam3_D_parts_nouns_norobot_split --num-shards 8 --shard-id 0
"""

import argparse
import glob
import json
import os
import time
import traceback

from run_sam3 import load_video
from sam3_batch_core import BG_COLORS, ROBOT_PROMPTS, load_model, run_session, write_outputs

DEFAULT_DATA_GLOB = ("/rlwrld-foundry-artifacts/.storage1/sjw_dataset/dataset/"
                     "PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/gr1_unified.*")
VIDEO_SUBPATH = "chunk-000/observation.images.ego_view"
HERE = os.path.dirname(os.path.abspath(__file__))


def dataset_fps(d):
    """Authoritative fps for a lerobot dataset dir (its meta/info.json)."""
    with open(os.path.join(d, "meta", "info.json")) as f:
        return float(json.load(f)["fps"])


def list_episodes(data_glob, nouns_path, max_nouns):
    """Fixed global ordering of (dataset_dir, episode_index, video_path, task, nouns, fps)."""
    with open(nouns_path) as f:
        task_nouns = json.load(f)

    episodes = []
    for d in sorted(glob.glob(data_glob)):
        fps = dataset_fps(d)
        tasks = {}
        with open(os.path.join(d, "meta", "episodes.jsonl")) as f:
            for line in f:
                j = json.loads(line)
                # strip the "unlocked_waist: " style prefix, as the noun table is keyed
                # on the bare instruction
                tasks[j["episode_index"]] = j["tasks"][0].split(": ", 1)[-1]
        for v in sorted(glob.glob(os.path.join(d, "videos", VIDEO_SUBPATH, "*.mp4"))):
            ep = int(os.path.basename(v).split("episode_")[-1].split(".")[0])
            task = tasks.get(ep, "")
            episodes.append((d, ep, v, task, task_nouns.get(task, [])[:max_nouns], fps))
    return episodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-glob", default=DEFAULT_DATA_GLOB)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--nouns", default=os.path.join(HERE, "task_nouns.json"))
    ap.add_argument("--model", default="jetjodh/sam3")
    ap.add_argument("--max-nouns", type=int, default=3,
                    help="cap on object prompts per episode (each costs ~0.04 s/frame)")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="process at most N episodes (smoke test)")
    ap.add_argument("--episodes", type=int, nargs="*", default=None,
                    help="explicit episode indices (ignores sharding; for smoke tests)")
    ap.add_argument("--cutout-from", default="union", choices=["union", "robot", "object"],
                    help="which mask the cutout video keeps (default: union of all prompts)")
    ap.add_argument("--bg", default="green", choices=sorted(BG_COLORS),
                    help="cutout background; green keeps masked-out area distinct from the black GR-1 hands")
    ap.add_argument("--no-prompt-masks", action="store_true",
                    help="drop the per-prompt planes from the npz (keeps mask/robot_mask/object_mask)")
    ap.add_argument("--fps", type=float, default=None,
                    help="override the fps stamped into the npz/videos; default = each "
                         "dataset's meta/info.json fps (20.0 for gr1_unified). load_video's "
                         "return value is hardcoded to 20.0 and must not be trusted.")
    ap.add_argument("--skip-videos", action="store_true",
                    help="write only the npz; reuse the cutout/overlay mp4s of the previous run")
    args = ap.parse_args()

    device, dtype = "cuda", __import__("torch").bfloat16
    bg_color = BG_COLORS[args.bg]

    episodes = list_episodes(args.data_glob, args.nouns, args.max_nouns)
    if args.episodes:
        wanted = set(args.episodes)
        mine = [e for e in episodes if e[1] in wanted]
    else:
        mine = [e for j, e in enumerate(episodes) if j % args.num_shards == args.shard_id]
    if args.limit:
        mine = mine[: args.limit]
    print(f"shard {args.shard_id}/{args.num_shards}: {len(mine)}/{len(episodes)} episodes", flush=True)

    model, processor = load_model(args.model, device, dtype)

    os.makedirs(args.out_root, exist_ok=True)
    manifest = os.path.join(args.out_root, f"manifest_shard{args.shard_id}.jsonl")
    n_done = n_skip = n_err = 0
    for d, ep, video_path, task, nouns, fps in mine:
        ds_name = os.path.basename(d.rstrip("/"))
        base = os.path.join(args.out_root, ds_name)
        stem = f"episode_{ep:06d}"
        cut_path = os.path.join(base, "cutout", VIDEO_SUBPATH, stem + ".mp4")
        ovl_path = os.path.join(base, "overlay", VIDEO_SUBPATH, stem + ".mp4")
        msk_path = os.path.join(base, "masks", VIDEO_SUBPATH, stem + ".npz")
        needed = [msk_path] if args.skip_videos else [cut_path, ovl_path, msk_path]
        if all(os.path.exists(p) for p in needed):
            n_skip += 1
            continue
        for p in needed:
            os.makedirs(os.path.dirname(p), exist_ok=True)

        prompts = ROBOT_PROMPTS + list(nouns)
        n_robot = len(ROBOT_PROMPTS)
        if args.fps is not None:
            fps = args.fps
        t0 = time.time()
        try:
            frames, _ = load_video(video_path)  # fps from info.json, not the loader
            prompt_masks = run_session(model, processor, frames, prompts, device, dtype)
            stats = write_outputs(
                frames, fps, prompt_masks, prompts, n_robot, ep,
                cut_path, ovl_path, msk_path, bg_color,
                cutout_from=args.cutout_from,
                keep_prompt_masks=not args.no_prompt_masks,
                write_videos=not args.skip_videos,
            )
            rec = {"dataset": ds_name, "episode_index": ep, "camera": VIDEO_SUBPATH,
                   "task": task, **stats, "seconds": round(time.time() - t0, 1)}
            n_done += 1
        except Exception as e:
            rec = {"dataset": ds_name, "episode_index": ep, "task": task,
                   "error": f"{type(e).__name__}: {e}"}
            traceback.print_exc()
            n_err += 1
        with open(manifest, "a") as f:
            f.write(json.dumps(rec) + "\n")
        done = n_done + n_skip + n_err
        msg = rec.get("error") or (f"{rec['frames_with_mask']}/{rec['n_frames']} frames "
                                   f"(robot {rec['robot_frames_with_mask']}, "
                                   f"object {rec['object_frames_with_mask']}), {rec['seconds']}s")
        print(f"[{done}/{len(mine)}] {ds_name}/ep{ep}: {msg}", flush=True)

    print(f"shard {args.shard_id} finished: done={n_done} skip={n_skip} err={n_err}", flush=True)


if __name__ == "__main__":
    main()
