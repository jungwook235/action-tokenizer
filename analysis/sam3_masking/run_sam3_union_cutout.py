"""SAM3 video-mode union cutout: union of "Robot" mask and task-instruction mask.

For each episode:
  - runs Sam3VideoModel with prompts ["Robot", <task instruction>]
  - writes cutout mp4 (union mask keeps pixels, rest = bg color) + orig|cutout
  - saves mask info to masks.npz + masks_meta.json
      <label>_idmask : (T, H, W) uint8, 0 = background, k = object slot k
                       (slot -> object_id mapping in masks_meta.json)
      union          : (T, H, W) uint8 binary union over both prompts
"""

import argparse
import json
import os

import numpy as np
import torch
from transformers import Sam3VideoModel, Sam3VideoProcessor

from run_sam3 import load_video, run_video_mode, write_video


def extract_frames_info(per_frame_raw, n_frames, shape):
    """-> idmask (T,H,W) uint8, meta list per frame."""
    idmask = np.zeros((n_frames,) + shape, np.uint8)
    meta = []
    for fi in range(n_frames):
        res = per_frame_raw.get(fi)
        fmeta = {"object_ids": [], "scores": [], "boxes": []}
        if res is not None and res.get("masks") is not None and len(res["masks"]):
            m = res["masks"]
            m = m.cpu().numpy() if hasattr(m, "cpu") else np.asarray(m)
            if m.ndim == 4:
                m = m[:, 0]
            obj_ids = res.get("object_ids", range(len(m)))
            obj_ids = obj_ids.tolist() if hasattr(obj_ids, "tolist") else list(obj_ids)
            scores = res.get("scores")
            scores = scores.float().cpu().numpy().tolist() if hasattr(scores, "cpu") else list(scores or [])
            boxes = res.get("boxes")
            boxes = boxes.float().cpu().numpy().tolist() if hasattr(boxes, "cpu") else list(boxes or [])
            # draw low-score first so high-score objects overwrite on overlap
            order = np.argsort(scores) if len(scores) == len(m) else range(len(m))
            for k in order:
                idmask[fi][m[k].astype(bool)] = int(obj_ids[k]) + 1
            fmeta = {"object_ids": [int(o) for o in obj_ids], "scores": scores, "boxes": boxes}
        meta.append(fmeta)
    return idmask, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--task", required=True, help="task instruction text prompt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="jetjodh/sam3")
    ap.add_argument("--bg", default="green", choices=["black", "white", "green"])
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda"
    dtype = torch.bfloat16
    bg_color = {"black": (0, 0, 0), "white": (255, 255, 255), "green": (0, 177, 64)}[args.bg]

    frames, fps = load_video(args.video)
    shape = frames[0].shape[:2]
    print(f"video: {args.video} ({len(frames)} frames)")

    model = Sam3VideoModel.from_pretrained(args.model, dtype=dtype).to(device).eval()
    processor = Sam3VideoProcessor.from_pretrained(args.model)

    prompts = {"robot": "Robot", "task": args.task}
    idmasks, metas = {}, {}
    for label, prompt in prompts.items():
        print(f"[video] prompt={prompt!r}")
        per_frame_raw, dt = run_video_mode(model, processor, frames, prompt, device, dtype)
        idmasks[label], metas[label] = extract_frames_info(per_frame_raw, len(frames), shape)
        print(f"  done in {dt:.1f}s, frames with mask: {(idmasks[label].reshape(len(frames), -1).any(1)).sum()}/{len(frames)}")

    union = ((idmasks["robot"] > 0) | (idmasks["task"] > 0)).astype(np.uint8)

    cutouts, combos = [], []
    for fi in range(len(frames)):
        cut = np.empty_like(frames[fi])
        cut[:] = bg_color
        keep = union[fi].astype(bool)
        cut[keep] = frames[fi][keep]
        cutouts.append(cut)
        combos.append(np.concatenate([frames[fi], cut], axis=1))
    write_video(os.path.join(args.out, "cutout_union.mp4"), cutouts, fps)
    write_video(os.path.join(args.out, "orig_vs_cutout_union.mp4"), combos, fps)

    np.savez_compressed(
        os.path.join(args.out, "masks.npz"),
        robot_idmask=idmasks["robot"],
        task_idmask=idmasks["task"],
        union=union,
    )
    with open(os.path.join(args.out, "masks_meta.json"), "w") as f:
        json.dump(
            {
                "video": args.video,
                "prompts": prompts,
                "n_frames": len(frames),
                "idmask_convention": "0=background, k=object_id k-1 of that prompt",
                "per_frame": {label: metas[label] for label in prompts},
            },
            f,
        )
    print("wrote", args.out, "(cutout_union.mp4, orig_vs_cutout_union.mp4, masks.npz, masks_meta.json)")


if __name__ == "__main__":
    main()
