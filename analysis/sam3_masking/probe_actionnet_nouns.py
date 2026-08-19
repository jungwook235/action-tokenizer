"""Fast image-mode screening of candidate SAM3 prompts on ActionNet episodes.

Mirrors probe_dexjoco_nouns.py: sample N frames evenly from an episode, run
image-mode Sam3Model once per candidate phrase, report frames-detected / mean
mask area / max score. ~1s per candidate, so a dozen candidates per episode is
cheap compared to a full video-mode pass.

Usage:
  python probe_actionnet_nouns.py --episodes 22131 23736 \
      --candidates "cabinet door" "wooden box" "roll top box" ...
"""

import argparse
import json
import os

import numpy as np
import torch

from run_sam3 import load_video, run_image_mode

DEFAULT_ROOT = "/rlwrld-foundry-artifacts/.storage1/sjw_dataset/dataset/ActionNet/gr1_actionnet_lerobot_15fps"
CAMERA = "observation.images.ego_view"
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--episodes", type=int, nargs="+", required=True)
    ap.add_argument("--candidates", nargs="+", required=True)
    ap.add_argument("--n-frames", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--model", default="jetjodh/sam3")
    ap.add_argument("--out", default=os.path.join(HERE, "outputs_actionnet", "probe_nouns.jsonl"))
    args = ap.parse_args()

    with open(os.path.join(args.root, "meta", "info.json")) as f:
        chunks_size = json.load(f)["chunks_size"]
    tasks = {}
    with open(os.path.join(args.root, "meta", "episodes.jsonl")) as f:
        for line in f:
            j = json.loads(line)
            tasks[j["episode_index"]] = j["tasks"][0]

    device, dtype = "cuda", torch.bfloat16
    from transformers import Sam3Model, Sam3Processor
    model = Sam3Model.from_pretrained(args.model, dtype=dtype).to(device).eval()
    processor = Sam3Processor.from_pretrained(args.model)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for ep in args.episodes:
        v = os.path.join(args.root, "videos", f"chunk-{ep // chunks_size:03d}", CAMERA,
                         f"episode_{ep:06d}.mp4")
        frames, _ = load_video(v)
        idxs = np.linspace(0, len(frames) - 1, args.n_frames).astype(int)
        sub = [frames[i] for i in idxs]
        print(f"\n=== ep{ep:06d} ({len(frames)} frames, sampled {list(idxs)}) :: {tasks[ep]}")
        for c in args.candidates:
            masks, scores, _ = run_image_mode(model, processor, sub, c, device, args.threshold)
            det = sum(1 for m in masks if len(m))
            areas = [m.any(0).mean() * 100 for m in masks if len(m)]
            smax = max((float(s.max()) for s in scores if len(s)), default=0.0)
            rec = {"episode_index": ep, "task": tasks[ep], "prompt": c,
                   "frames_detected": det, "n_frames": len(sub),
                   "mean_area_pct": round(float(np.mean(areas)), 2) if areas else 0.0,
                   "max_score": round(smax, 3)}
            with open(args.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"    {c:24s} det {det}/{len(sub)}  area {rec['mean_area_pct']:6.2f}%  "
                  f"max_score {smax:.2f}", flush=True)


if __name__ == "__main__":
    main()
