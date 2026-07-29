"""SAM3 video-mode cutout: keep only masked pixels, blank out the rest.

Writes per-prompt cutout mp4s (black background) plus original|cutout
side-by-side videos.
"""

import argparse
import os

import numpy as np
import torch
from transformers import Sam3VideoModel, Sam3VideoProcessor

from run_sam3 import load_video, run_video_mode, write_video


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--prompts", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--out", default="outputs_cutout")
    ap.add_argument("--model", default="jetjodh/sam3")
    ap.add_argument("--bg", default="black", choices=["black", "white", "green"])
    args = ap.parse_args()
    assert len(args.prompts) == len(args.labels)

    os.makedirs(args.out, exist_ok=True)
    device = "cuda"
    dtype = torch.bfloat16
    bg_color = {"black": (0, 0, 0), "white": (255, 255, 255), "green": (0, 177, 64)}[args.bg]

    frames, fps = load_video(args.video)
    print(f"video: {args.video} ({len(frames)} frames)")

    model = Sam3VideoModel.from_pretrained(args.model, dtype=dtype).to(device).eval()
    processor = Sam3VideoProcessor.from_pretrained(args.model)

    for prompt, label in zip(args.prompts, args.labels):
        print(f"[video] prompt={prompt!r}")
        per_frame_raw, dt = run_video_mode(model, processor, frames, prompt, device, dtype)
        cutouts, combos = [], []
        for fi in range(len(frames)):
            union = np.zeros(frames[fi].shape[:2], bool)
            res = per_frame_raw.get(fi)
            if res is not None and res.get("masks") is not None and len(res["masks"]):
                m = res["masks"]
                m = m.cpu().numpy() if hasattr(m, "cpu") else np.asarray(m)
                if m.ndim == 4:
                    m = m[:, 0]
                union = m.astype(bool).any(axis=0)
            cut = np.empty_like(frames[fi])
            cut[:] = bg_color
            cut[union] = frames[fi][union]
            cutouts.append(cut)
            combos.append(np.concatenate([frames[fi], cut], axis=1))
        write_video(os.path.join(args.out, f"cutout_{label}.mp4"), cutouts, fps)
        write_video(os.path.join(args.out, f"orig_vs_cutout_{label}.mp4"), combos, fps)
        print(f"  wrote cutout_{label}.mp4 / orig_vs_cutout_{label}.mp4 ({dt:.1f}s)")


if __name__ == "__main__":
    main()
