"""SAM3 text-prompt masking on gr1_actionnet_lerobot_15fps — sample-level probe.

Same machinery as compare_prompt_combos.py (gr1_unified), adapted to the ActionNet
layout: one flat lerobot dataset with 30 chunks (chunks_size=1000), single camera
`observation.images.ego_view` (256x256, letterboxed real teleop footage — two black
GR-1 dexterous arms entering from the left/right, humans sometimes visible in the
background).

There is no curated noun vocabulary here (1562 distinct task strings), so the
object-side prompt is the task instruction itself.

Combos:
  A_task        : ["Robot", <task>]                                (production default on gr1_unified)
  B_parts_task  : ["Robot", "robot hand", "robot arm", <task>]
  D_parts       : ["robot hand", "robot arm"]
  E_robot       : ["Robot"]
  F_black_parts : ["black robotic hand", "black robotic arm"]
  G_robot_parts : ["Robot", "robot hand", "robot arm"]        (B minus the dead task prompt)
  H_parts_nouns : ["robot hand", "robot arm", *nouns]         (nouns from --nouns json, keyed by task)
  I_nouns       : [*nouns]                                     (objects only, for per-noun diagnosis)

Outputs (out-dir default outputs_actionnet/):
  ep{ep:06d}_{combo}.mp4   overlay colored per prompt
  results.jsonl            one record per (episode, combo)

Usage:
  PYTHONPATH=venv_sam3/lib/python3.10/site-packages \
    ~/miniconda3/envs/gr00t/bin/python compare_prompt_combos_actionnet.py --n-episodes 6
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from compare_prompt_combos import overlay_by_prompt, run_combo, stats
from run_sam3 import load_video, write_video

DEFAULT_ROOT = "/rlwrld-foundry-artifacts/.storage1/sjw_dataset/dataset/ActionNet/gr1_actionnet_lerobot_15fps"
CAMERA = "observation.images.ego_view"
HERE = os.path.dirname(os.path.abspath(__file__))


def build_combos(task, nouns=None):
    combos = {
        "A_task": ["Robot", task],
        "B_parts_task": ["Robot", "robot hand", "robot arm", task],
        "D_parts": ["robot hand", "robot arm"],
        "E_robot": ["Robot"],
        "F_black_parts": ["black robotic hand", "black robotic arm"],
        "G_robot_parts": ["Robot", "robot hand", "robot arm"],
    }
    if nouns:
        combos["H_parts_nouns"] = ["robot hand", "robot arm", *nouns]
        combos["I_nouns"] = list(nouns)
    return combos


def episode_video(root, ep, chunks_size):
    return os.path.join(root, "videos", f"chunk-{ep // chunks_size:03d}", CAMERA,
                        f"episode_{ep:06d}.mp4")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "outputs_actionnet"))
    ap.add_argument("--model", default="jetjodh/sam3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-episodes", type=int, default=6)
    ap.add_argument("--episodes", type=int, nargs="*", default=None,
                    help="explicit episode indices (overrides --n-episodes sampling)")
    ap.add_argument("--combos", nargs="*", default=None, help="subset of combo names to run")
    ap.add_argument("--nouns", default=None,
                    help="json {task: [noun, ...]}; enables the H_parts_nouns / I_nouns combos")
    args = ap.parse_args()

    with open(os.path.join(args.root, "meta", "info.json")) as f:
        chunks_size = json.load(f)["chunks_size"]
    eps = [json.loads(l) for l in open(os.path.join(args.root, "meta", "episodes.jsonl"))]
    by_idx = {e["episode_index"]: e for e in eps}

    task_nouns = {}
    if args.nouns:
        with open(args.nouns) as f:
            task_nouns = json.load(f)

    rng = np.random.default_rng(args.seed)
    if args.episodes:
        sel = [by_idx[e] for e in args.episodes]
    else:
        sel = [eps[i] for i in rng.choice(len(eps), args.n_episodes, replace=False)]

    device, dtype = "cuda", torch.bfloat16
    from transformers import Sam3VideoModel, Sam3VideoProcessor
    model = Sam3VideoModel.from_pretrained(args.model, dtype=dtype).to(device).eval()
    processor = Sam3VideoProcessor.from_pretrained(args.model)

    os.makedirs(args.out_dir, exist_ok=True)
    res_path = os.path.join(args.out_dir, "results.jsonl")
    summary = []

    for e in sel:
        ep, task = e["episode_index"], e["tasks"][0]
        frames, fps = load_video(episode_video(args.root, ep, chunks_size))
        print(f"\n=== ep{ep:06d} ({len(frames)} frames) :: {task}", flush=True)
        combos = build_combos(task, task_nouns.get(task))
        if args.combos:
            combos = {k: v for k, v in combos.items() if k in args.combos}
        for name, prompts in combos.items():
            t0 = time.time()
            pm = run_combo(model, processor, frames, prompts, device, dtype)
            st = stats(pm, prompts)
            write_video(os.path.join(args.out_dir, f"ep{ep:06d}_{name}.mp4"),
                        overlay_by_prompt(frames, pm), fps)
            rec = {"episode_index": ep, "combo": name, "prompts": prompts, "task": task,
                   "seconds": round(time.time() - t0, 1), **st}
            with open(res_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            summary.append(rec)
            pp = "  ".join(f"{p!r}:{s['frames_detected']}/{st['n_frames']}@{s['mean_area_pct']}%"
                           for p, s in st["per_prompt"].items())
            print(f"  {name:14s} union {st['union_frames_with_mask']}/{st['n_frames']} "
                  f"first@{st['union_first_frame']} area {st['union_mean_area_pct']}% "
                  f"({rec['seconds']}s)\n      {pp}", flush=True)

    print("\n=== summary (union frames_with_mask / n_frames | first detected frame) ===")
    by_ep = {}
    for r in summary:
        by_ep.setdefault(r["episode_index"], {})[r["combo"]] = r
    for ep, cs in by_ep.items():
        print(f"ep{ep:06d}")
        for c, r in sorted(cs.items()):
            print(f"   {c:14s} {r['union_frames_with_mask']}/{r['n_frames']} "
                  f"first@{r['union_first_frame']} area {r['union_mean_area_pct']}%")


if __name__ == "__main__":
    main()
