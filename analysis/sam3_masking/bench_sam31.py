"""Benchmark official SAM 3.1 (multiplex) vs our transformers SAM3 results.

For each episode: run 3 text prompts (one session each — official API allows a
single text concept per session), collect per-frame masks, save cutout +
comparison video (orig | SAM3 cutout | SAM3.1 cutout) + masks.npz + timing.
"""

import argparse
import json
import os
import time
import uuid

import numpy as np
import torch

from run_sam3 import PALETTE, load_video, overlay, write_video

CKPT = "/sjw_alinlab1/home/jungwook/.cache/huggingface/hub/models--jetjodh--sam3.1/snapshots/d094d562d62ccbf550215d6697d5eb6193bfab83/sam3.1_multiplex.pt"
BG = (0, 177, 64)


def start_session(pred, video):
    state = pred.model.init_state(resource_path=video)
    sid = str(uuid.uuid4())
    pred._all_inference_states[sid] = {
        "state": state, "session_id": sid,
        "start_time": time.time(), "last_use_time": time.time(),
    }
    return sid


def run_prompt(pred, video, prompt):
    sid = start_session(pred, video)
    t0 = time.time()
    pred.handle_request(dict(type="add_prompt", session_id=sid, frame_index=0, text=prompt))
    per_frame = {}
    for item in pred.handle_stream_request(
        dict(type="propagate_in_video", session_id=sid, propagation_direction="forward")
    ):
        per_frame[item["frame_index"]] = item["outputs"]
    dt = time.time() - t0
    pred.handle_request(dict(type="close_session", session_id=sid))
    return per_frame, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--prompts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sam3-npz", default=None, help="masks.npz from transformers sam3 run for side-by-side")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    from sam3.model_builder import build_sam3_multiplex_video_predictor
    pred = build_sam3_multiplex_video_predictor(checkpoint_path=CKPT, use_fa3=False)

    frames, fps = load_video(args.video)
    n = len(frames)
    shape = frames[0].shape[:2]
    print(f"video: {args.video} ({n} frames)")

    idmasks = {}
    timing = {}
    meta = {"per_prompt": {}}
    for pi, prompt in enumerate(args.prompts):
        per_frame, dt = run_prompt(pred, args.video, prompt)
        timing[prompt] = round(dt, 1)
        idm = np.zeros((n,) + shape, np.uint8)
        pmeta = []
        for fi in range(n):
            o = per_frame.get(fi)
            fm = {"object_ids": [], "probs": [], "boxes": []}
            if o is not None and len(o["out_obj_ids"]):
                order = np.argsort(o["out_probs"])
                for k in order:
                    idm[fi][o["out_binary_masks"][k].astype(bool)] = int(o["out_obj_ids"][k]) + 1
                fm = {
                    "object_ids": [int(x) for x in o["out_obj_ids"]],
                    "probs": [float(x) for x in o["out_probs"]],
                    "boxes": np.asarray(o["out_boxes_xywh"]).tolist(),
                }
            pmeta.append(fm)
        idmasks[prompt] = idm
        meta["per_prompt"][prompt] = pmeta
        cov = int(idm.reshape(n, -1).any(1).sum())
        print(f"  {prompt!r}: {dt:.1f}s, frames with mask {cov}/{n}")

    union = np.zeros((n,) + shape, np.uint8)
    for idm in idmasks.values():
        union |= (idm > 0).astype(np.uint8)

    sam3_union = None
    if args.sam3_npz and os.path.exists(args.sam3_npz):
        sam3_union = np.load(args.sam3_npz)["union"]

    cutouts, combos = [], []
    for fi in range(n):
        cut = np.empty_like(frames[fi])
        cut[:] = BG
        keep = union[fi].astype(bool)
        cut[keep] = frames[fi][keep]
        cutouts.append(cut)
        panels = [frames[fi]]
        if sam3_union is not None:
            c3 = np.empty_like(frames[fi]); c3[:] = BG
            k3 = sam3_union[fi].astype(bool); c3[k3] = frames[fi][k3]
            panels.append(c3)
        panels.append(cut)
        combos.append(np.concatenate(panels, axis=1))
    write_video(os.path.join(args.out, "cutout_union_sam31.mp4"), cutouts, fps)
    label = "orig_sam3_sam31.mp4" if sam3_union is not None else "orig_sam31.mp4"
    write_video(os.path.join(args.out, label), combos, fps)

    np.savez_compressed(
        os.path.join(args.out, "masks_sam31.npz"),
        union=union,
        **{f"idmask_{i}": idmasks[p] for i, p in enumerate(args.prompts)},
    )
    meta.update({
        "video": args.video, "prompts": list(args.prompts), "n_frames": n,
        "timing_seconds": timing, "model": "sam3.1_multiplex (official repo)",
    })
    with open(os.path.join(args.out, "masks_meta_sam31.json"), "w") as f:
        json.dump(meta, f)
    print("timing:", timing, "total:", round(sum(timing.values()), 1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
