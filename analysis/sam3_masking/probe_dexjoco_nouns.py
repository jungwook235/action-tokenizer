"""Probe candidate SAM3 text prompts for dexjoco objects that got 0 detections.

For each (task, object) with a list of candidate phrases, sample N frames evenly
from the SAME episode used in compare_prompt_combos_dexjoco (episode index read
from outputs_dexjoco/results.jsonl), run image-mode Sam3Model per candidate, and
report frames-detected / mean mask area / max score. Fast: ~1s per candidate.

Usage: python probe_dexjoco_nouns.py [--n-frames 8] [--threshold 0.5]
"""

import argparse
import glob
import json
import os

import numpy as np
import torch

from run_sam3 import load_video, run_image_mode

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/sjw_alinlab1/home/jungwook/dataset/dexjoco_lerobot/v20"

# task -> {failed_object: [candidate prompts]} ("original" first for reference)
CANDIDATES = {
    "bimanual_assembly": {
        "tray": ["tray", "blue tray", "blue stand", "blue block", "blue bracket",
                 "gray block", "metal fixture", "blue container"],
        "peg": ["peg", "yellow peg", "yellow stick", "yellow rod", "yellow cylinder", "stick"],
    },
    "bimanual_hanoi": {
        "disk": ["disk", "colored block", "toy block", "red block", "blue block",
                 "yellow block", "wooden toy", "stacking toy"],
        "peg": ["peg", "wooden peg", "wooden stick", "wooden rod", "wooden pole", "wooden board"],
    },
    "fold_glasses": {
        "glasses case": ["glasses case", "wooden box", "brown box", "open box",
                         "cardboard box", "box", "wooden case"],
    },
    "pick_bucket": {
        "bucket": ["bucket", "red basket", "basket", "shopping basket",
                   "plastic basket", "red container", "red crate"],
        "food box": ["food box", "red box", "small box", "box", "food package",
                     "carton", "snack box"],
    },
    "pinch_tongs": {
        "tongs": ["tongs", "red tongs", "kitchen tongs", "red pliers",
                  "red clamp", "red tool", "red gripper"],
    },
    "water_plant": {
        "watering can": ["watering can", "red watering can", "red can", "red bottle",
                         "red cup", "red pitcher", "red container", "red cylinder"],
    },
}


def episode_videos_from_results():
    """(task -> video path) for the episodes used in the combo run."""
    out = {}
    with open(os.path.join(HERE, "outputs_dexjoco", "results.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            ds, ep, cam = r["dataset"], r["episode_index"], r["camera"]
            d = os.path.join(ROOT, ds) if os.path.isdir(os.path.join(ROOT, ds)) \
                else os.path.join(ROOT, "bimanual", ds)
            out[ds] = os.path.join(d, "videos", "chunk-000", cam, f"episode_{ep:06d}.mp4")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-frames", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--model", default="jetjodh/sam3")
    args = ap.parse_args()

    device, dtype = "cuda", torch.bfloat16
    from transformers import Sam3Model, Sam3Processor
    model = Sam3Model.from_pretrained(args.model, dtype=dtype).to(device).eval()
    processor = Sam3Processor.from_pretrained(args.model)

    videos = episode_videos_from_results()
    res_path = os.path.join(HERE, "outputs_dexjoco", "probe_nouns.jsonl")
    for task, objs in CANDIDATES.items():
        frames, _ = load_video(videos[task])
        idxs = np.linspace(0, len(frames) - 1, args.n_frames).astype(int)
        sub = [frames[i] for i in idxs]
        print(f"\n=== {task} ({os.path.basename(videos[task])}, frames {list(idxs)}) ===")
        for obj, cands in objs.items():
            print(f"  [{obj}]")
            for c in cands:
                masks, scores, _ = run_image_mode(model, processor, sub, c, device, args.threshold)
                det = sum(1 for m in masks if len(m))
                areas = [m.any(0).mean() * 100 for m in masks if len(m)]
                smax = max((float(s.max()) for s in scores if len(s)), default=0.0)
                nobj = [len(m) for m in masks]
                rec = {"task": task, "object": obj, "prompt": c,
                       "frames_detected": det, "n_frames": len(sub),
                       "mean_area_pct": round(float(np.mean(areas)), 2) if areas else 0.0,
                       "max_score": round(smax, 3), "n_objects_per_frame": nobj}
                with open(res_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"    {c:22s} det {det}/{len(sub)}  area {rec['mean_area_pct']:5.2f}%  "
                      f"max_score {smax:.2f}  nobj {nobj}", flush=True)


if __name__ == "__main__":
    main()
