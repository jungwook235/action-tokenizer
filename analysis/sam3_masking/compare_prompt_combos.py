"""Compare SAM3 text-prompt combos on one random episode per gr1_unified dataset.

Combos ("Robot" always included):
  A_task            : ["Robot", <task instruction>]                       (production default)
  B_parts_task      : ["Robot", "robot hand", "robot arm", <task>]
  C_parts_nouns     : ["Robot", "robot hand", "robot arm", *nouns]        (nouns from task_nouns.json)

Per dataset one episode is drawn with a fixed seed -> same episode across combos.
Outputs (out-dir default outputs_prompt_combos/):
  <ds_short>/ep{ep:06d}_{combo}.mp4      overlay, colored PER PROMPT (same prompt idx = same color
                                         across combos; union of a prompt's objects)
  results.jsonl                          one record per (dataset, combo) with per-prompt stats
  summary printed at the end

Stats per prompt: frames with >=1 detection, mean mask area (% of pixels, over detected frames).
Union stats: frames_with_mask, first/last detected frame, mean area.
"""

import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from run_sam3 import PALETTE, load_video, write_video

DEFAULT_DATA_GLOB = "/storage1/sjw_dataset/dataset/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/gr1_unified.*"
VIDEO_SUBPATH = "chunk-000/observation.images.ego_view"
HERE = os.path.dirname(os.path.abspath(__file__))


def build_combos(task, nouns):
    return {
        "A_task": ["Robot", task],
        "B_parts_task": ["Robot", "robot hand", "robot arm", task],
        "C_parts_nouns": ["Robot", "robot hand", "robot arm", *nouns],
        "D_parts_nouns_norobot": ["robot hand", "robot arm", *nouns],
    }


def run_combo(model, processor, frames, prompts, device, dtype):
    n = len(frames)
    h, w = frames[0].shape[:2]
    session = processor.init_video_session(
        video=frames, inference_device=device, video_storage_device="cpu", dtype=dtype
    )
    processor.add_text_prompt(session, text=prompts)
    per_frame = {}
    with torch.inference_mode():
        for out in model.propagate_in_video_iterator(session):
            per_frame[out.frame_idx] = processor.postprocess_outputs(session, out)
    del session

    # per-prompt binary masks per frame
    prompt_masks = np.zeros((len(prompts), n, h, w), np.uint8)
    for fi in range(n):
        res = per_frame.get(fi)
        if res is None or res.get("masks") is None or not len(res["masks"]):
            continue
        m = res["masks"]
        m = m.cpu().numpy() if hasattr(m, "cpu") else np.asarray(m)
        if m.ndim == 4:
            m = m[:, 0]
        oi = res["object_ids"]
        oi = oi.tolist() if hasattr(oi, "tolist") else list(oi)
        obj2prompt = {}
        for ptxt, oids in res.get("prompt_to_obj_ids", {}).items():
            pi = prompts.index(ptxt) if ptxt in prompts else None
            for o in oids:
                obj2prompt[int(o)] = pi
        for mm, o in zip(m, oi):
            pi = obj2prompt.get(int(o))
            if pi is None:
                continue
            prompt_masks[pi, fi][mm.astype(bool)] = 1
    return prompt_masks


def overlay_by_prompt(frames, prompt_masks, alpha=0.55):
    out = []
    for fi, fr in enumerate(frames):
        o = fr.astype(np.float32)
        for pi in range(prompt_masks.shape[0]):
            mm = prompt_masks[pi, fi].astype(bool)
            if mm.any():
                color = PALETTE[pi % len(PALETTE)].astype(np.float32)
                o[mm] = o[mm] * (1 - alpha) + color * alpha
        out.append(o.astype(np.uint8))
    return out


def stats(prompt_masks, prompts):
    n = prompt_masks.shape[1]
    per_prompt = {}
    for pi, p in enumerate(prompts):
        pm = prompt_masks[pi]
        cov = pm.reshape(n, -1).any(1)
        det = int(cov.sum())
        per_prompt[p] = {
            "frames_detected": det,
            "mean_area_pct": round(float(pm[cov].mean() * 100), 2) if det else 0.0,
        }
    union = prompt_masks.any(0)
    ucov = union.reshape(n, -1).any(1)
    idx = np.where(ucov)[0]
    return {
        "n_frames": n,
        "union_frames_with_mask": int(ucov.sum()),
        "union_first_frame": int(idx[0]) if len(idx) else None,
        "union_last_frame": int(idx[-1]) if len(idx) else None,
        "union_mean_area_pct": round(float(union[ucov].mean() * 100), 2) if len(idx) else 0.0,
        "per_prompt": per_prompt,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-glob", default=DEFAULT_DATA_GLOB)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "outputs_prompt_combos"))
    ap.add_argument("--model", default="jetjodh/sam3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--combos", nargs="*", default=None, help="subset of combo names to run")
    ap.add_argument("--limit-datasets", type=int, default=None)
    args = ap.parse_args()

    with open(os.path.join(HERE, "task_nouns.json")) as f:
        task_nouns = json.load(f)

    device, dtype = "cuda", torch.bfloat16
    from transformers import Sam3VideoModel, Sam3VideoProcessor
    model = Sam3VideoModel.from_pretrained(args.model, dtype=dtype).to(device).eval()
    processor = Sam3VideoProcessor.from_pretrained(args.model)

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    res_path = os.path.join(args.out_dir, "results.jsonl")
    summary = []

    dirs = sorted(glob.glob(args.data_glob))
    if args.limit_datasets:
        dirs = dirs[: args.limit_datasets]
    for d in dirs:
        ds = os.path.basename(d.rstrip("/"))
        ds_short = ds.split(".", 1)[-1].replace("_GR1ArmsAndWaistFourierHands_1000", "")
        tasks = {}
        with open(os.path.join(d, "meta", "episodes.jsonl")) as f:
            for line in f:
                j = json.loads(line)
                tasks[j["episode_index"]] = j["tasks"][0].split(": ", 1)[-1]
        vids = sorted(glob.glob(os.path.join(d, "videos", VIDEO_SUBPATH, "*.mp4")))
        v = vids[int(rng.integers(len(vids)))]
        ep = int(os.path.basename(v).split("episode_")[-1].split(".")[0])
        task = tasks[ep]
        nouns = task_nouns[task]  # KeyError = vocab gap, fail loudly

        frames, fps = load_video(v)
        combos = build_combos(task, nouns)
        if args.combos:
            combos = {k: v2 for k, v2 in combos.items() if k in args.combos}
        os.makedirs(os.path.join(args.out_dir, ds_short), exist_ok=True)
        for name, prompts in combos.items():
            t0 = time.time()
            pm = run_combo(model, processor, frames, prompts, device, dtype)
            st = stats(pm, prompts)
            write_video(
                os.path.join(args.out_dir, ds_short, f"ep{ep:06d}_{name}.mp4"),
                overlay_by_prompt(frames, pm), fps,
            )
            rec = {"dataset": ds, "episode_index": ep, "combo": name, "prompts": prompts,
                   "task": task, "seconds": round(time.time() - t0, 1), **st}
            with open(res_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            summary.append(rec)
            print(f"{ds_short}/ep{ep} {name}: union {st['union_frames_with_mask']}/{st['n_frames']} "
                  f"first@{st['union_first_frame']} area {st['union_mean_area_pct']}% "
                  f"({rec['seconds']}s)", flush=True)

    print("\n=== summary (union frames_with_mask / n_frames | first detected frame) ===")
    by_ds = {}
    for r in summary:
        by_ds.setdefault((r["dataset"], r["episode_index"]), {})[r["combo"]] = r
    for (ds, ep), cs in by_ds.items():
        row = " | ".join(
            f"{c}: {r['union_frames_with_mask']}/{r['n_frames']} first@{r['union_first_frame']}"
            for c, r in sorted(cs.items())
        )
        print(f"{ds.split('.',1)[-1][:52]:52s} ep{ep}: {row}")


if __name__ == "__main__":
    main()
