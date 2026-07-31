"""Batch SAM3 masking over GR-1 unified datasets (production, shardable).

Prompts per episode: ["Robot", <task instruction, "unlocked_waist: " prefix stripped>]
run in ONE Sam3VideoModel session (single propagation).

Outputs mirror the lerobot dataset layout so training code can look up by
(dataset dir name, episode_index, frame index == step index, fps 20):

  <out_root>/<dataset_dirname>/
      cutout/chunk-000/observation.images.ego_view/episode_XXXXXX.mp4   # masked pixels only, rest = bg
      overlay/chunk-000/observation.images.ego_view/episode_XXXXXX.mp4  # original + colored instance masks
      masks/chunk-000/observation.images.ego_view/episode_XXXXXX.npz    # mask: (T,H,W) uint8, 0/1 union
      manifest_shard{K}.jsonl                                           # per-episode log

Sharding: episodes are enumerated in a fixed sorted order across all dataset
dirs; episode j is processed by shard (j % num_shards). Existing outputs are
skipped, so shards are resumable / re-runnable.
"""

import argparse
import glob
import json
import os
import time
import traceback

import numpy as np
import torch

from run_sam3 import PALETTE, load_video, write_video

DEFAULT_DATA_GLOB = "/storage1/sjw_dataset/dataset/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/gr1_unified.*"
VIDEO_SUBPATH = "chunk-000/observation.images.ego_view"


def overlay_clean(frame_rgb, masks, ids, alpha=0.55):
    """Colored instance-mask overlay, no text/labels."""
    out = frame_rgb.astype(np.float32)
    for m, i in zip(masks, ids):
        color = PALETTE[i % len(PALETTE)].astype(np.float32)
        out[m] = out[m] * (1 - alpha) + color * alpha
    return out.astype(np.uint8)


def list_episodes(data_glob):
    """Fixed global ordering of (dataset_dir, episode_index, video_path, task)."""
    episodes = []
    for d in sorted(glob.glob(data_glob)):
        tasks = {}
        with open(os.path.join(d, "meta", "episodes.jsonl")) as f:
            for line in f:
                j = json.loads(line)
                tasks[j["episode_index"]] = j["tasks"][0]
        for v in sorted(glob.glob(os.path.join(d, "videos", VIDEO_SUBPATH, "*.mp4"))):
            ep = int(os.path.basename(v).split("episode_")[-1].split(".")[0])
            task = tasks.get(ep, "")
            task = task.split(": ", 1)[-1]  # strip "unlocked_waist: " style prefix
            episodes.append((d, ep, v, task))
    return episodes


BG_COLORS = {"black": (0, 0, 0), "white": (255, 255, 255), "green": (0, 177, 64)}


def process_episode(model, processor, video_path, task, device, dtype, bg_color):
    frames, fps = load_video(video_path)
    n = len(frames)
    shape = frames[0].shape[:2]
    prompts = ["Robot", task]

    session = processor.init_video_session(
        video=frames, inference_device=device, video_storage_device="cpu", dtype=dtype
    )
    processor.add_text_prompt(session, text=prompts)
    per_frame = {}
    with torch.inference_mode():
        for out in model.propagate_in_video_iterator(session):
            per_frame[out.frame_idx] = processor.postprocess_outputs(session, out)
    del session

    union = np.zeros((n,) + shape, np.uint8)
    cutouts, overlays = [], []
    frames_with_mask = 0
    for fi in range(n):
        res = per_frame.get(fi)
        masks_l, ids_l = [], []
        if res is not None and res.get("masks") is not None and len(res["masks"]):
            m = res["masks"]
            m = m.cpu().numpy() if hasattr(m, "cpu") else np.asarray(m)
            if m.ndim == 4:
                m = m[:, 0]
            masks_l = [mm.astype(bool) for mm in m]
            oi = res["object_ids"]
            ids_l = oi.tolist() if hasattr(oi, "tolist") else list(oi)
            for mm in masks_l:
                union[fi][mm] = 1
        if union[fi].any():
            frames_with_mask += 1
        cut = np.empty_like(frames[fi])
        cut[:] = bg_color
        keep = union[fi].astype(bool)
        cut[keep] = frames[fi][keep]
        cutouts.append(cut)
        overlays.append(overlay_clean(frames[fi], masks_l, ids_l))
    return frames, fps, union, cutouts, overlays, prompts, frames_with_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-glob", default=DEFAULT_DATA_GLOB)
    ap.add_argument("--out-root", default="/storage1/sjw_dataset/dataset/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim_sam3_robot_task")
    ap.add_argument("--model", default="jetjodh/sam3")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="process at most this many episodes (smoke test)")
    ap.add_argument("--bg", default="green", choices=sorted(BG_COLORS),
                    help="cutout background color; green (default) keeps masked-out area distinguishable from the black robot hands")
    args = ap.parse_args()

    device = "cuda"
    dtype = torch.bfloat16

    episodes = list_episodes(args.data_glob)
    mine = [e for j, e in enumerate(episodes) if j % args.num_shards == args.shard_id]
    if args.limit:
        mine = mine[: args.limit]
    print(f"shard {args.shard_id}/{args.num_shards}: {len(mine)}/{len(episodes)} episodes", flush=True)

    from transformers import Sam3VideoModel, Sam3VideoProcessor
    model = Sam3VideoModel.from_pretrained(args.model, dtype=dtype).to(device).eval()
    processor = Sam3VideoProcessor.from_pretrained(args.model)

    n_done = n_skip = n_err = 0
    for d, ep, video_path, task in mine:
        ds_name = os.path.basename(d.rstrip("/"))
        base = os.path.join(args.out_root, ds_name)
        stem = f"episode_{ep:06d}"
        cut_path = os.path.join(base, "cutout", VIDEO_SUBPATH, stem + ".mp4")
        ovl_path = os.path.join(base, "overlay", VIDEO_SUBPATH, stem + ".mp4")
        msk_path = os.path.join(base, "masks", VIDEO_SUBPATH, stem + ".npz")
        if os.path.exists(cut_path) and os.path.exists(ovl_path) and os.path.exists(msk_path):
            n_skip += 1
            continue
        for p in (cut_path, ovl_path, msk_path):
            os.makedirs(os.path.dirname(p), exist_ok=True)
        t0 = time.time()
        try:
            frames, fps, union, cutouts, overlays, prompts, cov = process_episode(
                model, processor, video_path, task, device, dtype, BG_COLORS[args.bg]
            )
            write_video(cut_path, cutouts, fps)
            write_video(ovl_path, overlays, fps)
            np.savez_compressed(
                msk_path,
                mask=union,  # (T, H, W) uint8 in {0,1}; frame t == step t of the episode parquet
                episode_index=ep,
                prompts=np.array(prompts),
                fps=fps,
            )
            rec = {
                "dataset": ds_name, "episode_index": ep, "n_frames": len(frames),
                "frames_with_mask": cov, "task": task, "seconds": round(time.time() - t0, 1),
            }
            n_done += 1
        except Exception as e:
            rec = {"dataset": ds_name, "episode_index": ep, "error": f"{type(e).__name__}: {e}"}
            traceback.print_exc()
            n_err += 1
        with open(os.path.join(args.out_root, f"manifest_shard{args.shard_id}.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{n_done+n_skip+n_err}/{len(mine)}] {ds_name}/ep{ep}: {rec}", flush=True)

    print(f"shard {args.shard_id} finished: done={n_done} skip={n_skip} err={n_err}", flush=True)


if __name__ == "__main__":
    main()
