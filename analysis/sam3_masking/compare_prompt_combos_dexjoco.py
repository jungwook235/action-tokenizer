"""SAM3 text-prompt masking on dexjoco_lerobot/v20 — one random episode per task.

Same machinery as compare_prompt_combos.py (gr1_unified), adapted to the dexjoco
v20 layout:
  v20/<task>/{meta,videos}                       (single-arm tasks)
  v20/bimanual/<task>/{meta,videos}              (bimanual tasks)
Camera per task is the first non-wrist view found in videos/chunk-000/
(preference: ego > ego_right > front).

Default combo: D_parts_nouns_norobot = ["robot hand", "robot arm", *nouns],
nouns taken from dexjoco_task_nouns.json keyed by the exact task string.

Outputs (out-dir default outputs_dexjoco/):
  <task>/ep{ep:06d}_{combo}.mp4   overlay colored per prompt
  results.jsonl                   one record per (task, combo)
"""

import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from compare_prompt_combos import build_combos, overlay_by_prompt, run_combo, stats
from run_sam3 import load_video, write_video

DEFAULT_ROOT = "/sjw_alinlab1/home/jungwook/dataset/dexjoco_lerobot/v20"
CAMERA_PREF = ["observation.images.ego", "observation.images.ego_right", "observation.images.front"]
HERE = os.path.dirname(os.path.abspath(__file__))


def find_task_dirs(root):
    """Yield (task_name, dataset_dir) for every dir under root that has meta/episodes.jsonl."""
    out = []
    for d in sorted(glob.glob(os.path.join(root, "*")) + glob.glob(os.path.join(root, "bimanual", "*"))):
        if os.path.isfile(os.path.join(d, "meta", "episodes.jsonl")):
            out.append((os.path.basename(d.rstrip("/")), d))
    return out


def pick_camera(d):
    cams = sorted(glob.glob(os.path.join(d, "videos", "chunk-000", "*")))
    names = [os.path.basename(c) for c in cams]
    for pref in CAMERA_PREF:
        if pref in names:
            return pref
    non_wrist = [n for n in names if "wrist" not in n]
    if not non_wrist:
        raise RuntimeError(f"no non-wrist camera in {d}: {names}")
    return non_wrist[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "outputs_dexjoco"))
    ap.add_argument("--model", default="jetjodh/sam3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--combos", nargs="*", default=["D_parts_nouns_norobot"])
    ap.add_argument("--limit-tasks", type=int, default=None)
    ap.add_argument("--tasks", nargs="*", default=None, help="subset of task dir names to run")
    args = ap.parse_args()

    with open(os.path.join(HERE, "dexjoco_task_nouns.json")) as f:
        task_nouns = json.load(f)

    device, dtype = "cuda", torch.bfloat16
    from transformers import Sam3VideoModel, Sam3VideoProcessor
    model = Sam3VideoModel.from_pretrained(args.model, dtype=dtype).to(device).eval()
    processor = Sam3VideoProcessor.from_pretrained(args.model)

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    res_path = os.path.join(args.out_dir, "results.jsonl")
    summary = []

    task_dirs = find_task_dirs(args.root)
    if args.tasks:
        task_dirs = [(t, d) for t, d in task_dirs if t in args.tasks]
    if args.limit_tasks:
        task_dirs = task_dirs[: args.limit_tasks]
    for ds, d in task_dirs:
        cam = pick_camera(d)
        tasks = {}
        with open(os.path.join(d, "meta", "episodes.jsonl")) as f:
            for line in f:
                j = json.loads(line)
                tasks[j["episode_index"]] = j["tasks"][0]
        vids = sorted(glob.glob(os.path.join(d, "videos", "chunk-000", cam, "*.mp4")))
        v = vids[int(rng.integers(len(vids)))]
        ep = int(os.path.basename(v).split("episode_")[-1].split(".")[0])
        task = tasks[ep]
        nouns = task_nouns[task]  # KeyError = vocab gap, fail loudly

        frames, fps = load_video(v)
        combos = build_combos(task, nouns)
        combos = {k: v2 for k, v2 in combos.items() if k in args.combos}
        os.makedirs(os.path.join(args.out_dir, ds), exist_ok=True)
        for name, prompts in combos.items():
            t0 = time.time()
            pm = run_combo(model, processor, frames, prompts, device, dtype)
            st = stats(pm, prompts)
            write_video(
                os.path.join(args.out_dir, ds, f"ep{ep:06d}_{name}.mp4"),
                overlay_by_prompt(frames, pm), fps,
            )
            rec = {"dataset": ds, "camera": cam, "episode_index": ep, "combo": name,
                   "prompts": prompts, "task": task, "seconds": round(time.time() - t0, 1), **st}
            with open(res_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            summary.append(rec)
            print(f"{ds}/ep{ep} {name}: union {st['union_frames_with_mask']}/{st['n_frames']} "
                  f"first@{st['union_first_frame']} area {st['union_mean_area_pct']}% "
                  f"({rec['seconds']}s)", flush=True)

    print("\n=== summary (union frames_with_mask / n_frames | first detected frame) ===")
    for r in summary:
        print(f"{r['dataset']:32s} ep{r['episode_index']} {r['combo']}: "
              f"{r['union_frames_with_mask']}/{r['n_frames']} first@{r['union_first_frame']}")


if __name__ == "__main__":
    main()
