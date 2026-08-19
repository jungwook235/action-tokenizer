"""Batch SAM3 masking over gr1_actionnet_lerobot_15fps (production, shardable).

Prompts per episode, all run in ONE Sam3VideoModel session (single propagation):

    ROBOT_PROMPTS + task_nouns[task]          e.g. ["robot hand", "robot arm",
                                                    "alarm clock", "white basket"]

The robot half is fixed; the object half comes from actionnet_task_nouns.json
(built by build_actionnet_task_nouns.py). Probe findings behind this choice, from
compare_prompt_combos_actionnet.py over 6 random episodes:
  * ["robot hand","robot arm"] covers both arms from frame 0 in 6/6 episodes;
    "Robot" alone silently drops the idle arm in 2/6 and starts 15-24 frames late.
  * the task instruction as a single prompt is dead here (0 detections in 5/6),
    but its NOUNS detect in ~100% of frames.

Outputs mirror the lerobot layout of the source dataset (multi-chunk), matching
the dexjoco_lerobot_v20_sam3_* layout so downstream code can look up by
(episode_index, frame index == step index):

  <out_root>/
      cutout/chunk-XXX/observation.images.ego_view/episode_XXXXXX.mp4
      overlay/chunk-XXX/observation.images.ego_view/episode_XXXXXX.mp4
      masks/chunk-XXX/observation.images.ego_view/episode_XXXXXX.npz
      manifest_shard{K}.jsonl

masks npz keys (uint8 0/1, frame t == step t of the episode parquet):
      mask          (T,H,W)    union of every prompt          [dexjoco-compatible]
      robot_mask    (T,H,W)    union of the ROBOT_PROMPTS only
      object_mask   (T,H,W)    union of the task-noun prompts only
      prompt_masks  (P,T,H,W)  one plane per prompt           [dexjoco-compatible]
      prompts       (P,)       the prompt strings, same order as prompt_masks
      prompt_roles  (P,)       "robot" / "object", parallel to prompts
      n_robot_prompts ()       prompt_masks[:n] is the robot group, [n:] the objects
      episode_index (), fps ()

Sharding: episodes are enumerated in sorted order; episode j goes to shard
(j % num_shards). Existing outputs are skipped, so shards are resumable and can
be re-run with a different shard count.

Example:
  PYTHONPATH=venv_sam3/lib/python3.10/site-packages \
    ~/miniconda3/envs/gr00t/bin/python batch_sam3_actionnet.py \
      --out-root /path/to/gr1_actionnet_sam3_parts_nouns --num-shards 8 --shard-id 0
"""

import argparse
import json
import os
import time
import traceback

import torch

from run_sam3 import load_video
from sam3_batch_core import BG_COLORS, ROBOT_PROMPTS, load_model, run_session, write_outputs

DEFAULT_ROOT = "/rlwrld-foundry-artifacts/.storage1/sjw_dataset/dataset/ActionNet/gr1_actionnet_lerobot_15fps"
CAMERA = "observation.images.ego_view"
HERE = os.path.dirname(os.path.abspath(__file__))

def list_episodes(root, nouns_path, max_nouns):
    """Fixed global ordering of (episode_index, chunk_name, video_path, task, nouns)."""
    with open(os.path.join(root, "meta", "info.json")) as f:
        chunks_size = json.load(f)["chunks_size"]
    with open(nouns_path) as f:
        task_nouns = json.load(f)

    episodes = []
    with open(os.path.join(root, "meta", "episodes.jsonl")) as f:
        for line in f:
            j = json.loads(line)
            ep, task = j["episode_index"], j["tasks"][0]
            chunk = f"chunk-{ep // chunks_size:03d}"
            video = os.path.join(root, "videos", chunk, CAMERA, f"episode_{ep:06d}.mp4")
            # missing entry = instruction outside the vocabulary; robot prompts still apply
            nouns = task_nouns.get(task, [])[:max_nouns]
            episodes.append((ep, chunk, video, task, nouns))
    episodes.sort(key=lambda e: e[0])
    return episodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--nouns", default=os.path.join(HERE, "actionnet_task_nouns.json"))
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
                    help="fps stamped into the npz and the written videos; "
                         "default = meta/info.json fps (15.0 here). Do NOT rely on the value "
                         "load_video returns: it is hardcoded to 20.0 and is wrong for this dataset.")
    args = ap.parse_args()

    device, dtype = "cuda", torch.bfloat16
    bg_color = BG_COLORS[args.bg]

    with open(os.path.join(args.root, "meta", "info.json")) as f:
        fps = args.fps if args.fps is not None else float(json.load(f)["fps"])
    print(f"fps stamped into outputs: {fps}", flush=True)

    episodes = list_episodes(args.root, args.nouns, args.max_nouns)
    if args.episodes:
        wanted = set(args.episodes)
        mine = [e for e in episodes if e[0] in wanted]
    else:
        mine = [e for j, e in enumerate(episodes) if j % args.num_shards == args.shard_id]
    if args.limit:
        mine = mine[: args.limit]
    print(f"shard {args.shard_id}/{args.num_shards}: {len(mine)}/{len(episodes)} episodes", flush=True)

    model, processor = load_model(args.model, device, dtype)

    manifest = os.path.join(args.out_root, f"manifest_shard{args.shard_id}.jsonl")
    os.makedirs(args.out_root, exist_ok=True)
    n_done = n_skip = n_err = 0
    for ep, chunk, video_path, task, nouns in mine:
        sub = os.path.join(chunk, CAMERA)
        stem = f"episode_{ep:06d}"
        cut_path = os.path.join(args.out_root, "cutout", sub, stem + ".mp4")
        ovl_path = os.path.join(args.out_root, "overlay", sub, stem + ".mp4")
        msk_path = os.path.join(args.out_root, "masks", sub, stem + ".npz")
        if all(os.path.exists(p) for p in (cut_path, ovl_path, msk_path)):
            n_skip += 1
            continue
        for p in (cut_path, ovl_path, msk_path):
            os.makedirs(os.path.dirname(p), exist_ok=True)

        prompts = ROBOT_PROMPTS + list(nouns)
        n_robot = len(ROBOT_PROMPTS)
        t0 = time.time()
        try:
            frames, _ = load_video(video_path)  # fps from info.json, not the loader
            prompt_masks = run_session(model, processor, frames, prompts, device, dtype)
            stats = write_outputs(
                frames, fps, prompt_masks, prompts, n_robot, ep,
                cut_path, ovl_path, msk_path, bg_color,
                cutout_from=args.cutout_from,
                keep_prompt_masks=not args.no_prompt_masks,
            )
            rec = {"episode_index": ep, "chunk": chunk, "camera": CAMERA, "task": task,
                   **stats, "seconds": round(time.time() - t0, 1)}
            n_done += 1
        except Exception as e:
            rec = {"episode_index": ep, "chunk": chunk, "task": task,
                   "error": f"{type(e).__name__}: {e}"}
            traceback.print_exc()
            n_err += 1
        with open(manifest, "a") as f:
            f.write(json.dumps(rec) + "\n")
        msg = rec.get("error") or (f"{rec['frames_with_mask']}/{rec['n_frames']} frames "
                                   f"(robot {rec['robot_frames_with_mask']}, "
                                   f"object {rec['object_frames_with_mask']}), {rec['seconds']}s")
        print(f"[{n_done + n_skip + n_err}/{len(mine)}] ep{ep:06d}: {msg}", flush=True)

    print(f"shard {args.shard_id} finished: done={n_done} skip={n_skip} err={n_err}", flush=True)


if __name__ == "__main__":
    main()
