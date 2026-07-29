"""SAM3 video-mode multi-prompt cutout: all prompts in ONE session/propagation.

Prompts (e.g. "Robot hand", "Robot arm", "<object>") are segmented jointly;
the union of all masks is cut out. Saves per-prompt id-masks + metadata.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from transformers import Sam3VideoModel, Sam3VideoProcessor

from run_sam3 import PALETTE, load_video, overlay, write_video


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--prompts", nargs="+", required=True)
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
    print(f"video: {args.video} ({len(frames)} frames), prompts: {args.prompts}")

    model = Sam3VideoModel.from_pretrained(args.model, dtype=dtype).to(device).eval()
    processor = Sam3VideoProcessor.from_pretrained(args.model)

    session = processor.init_video_session(
        video=frames, inference_device=device, video_storage_device="cpu", dtype=dtype
    )
    processor.add_text_prompt(session, text=list(args.prompts))

    t0 = time.time()
    per_frame = {}
    with torch.inference_mode():
        for out in model.propagate_in_video_iterator(session, show_progress_bar=True):
            per_frame[out.frame_idx] = processor.postprocess_outputs(session, out)
    print(f"propagation: {time.time() - t0:.1f}s")

    n = len(frames)
    # per-prompt idmask: 0=bg, k=object_id k-1 (object ids are global across prompts)
    idmasks = {p: np.zeros((n,) + shape, np.uint8) for p in args.prompts}
    union = np.zeros((n,) + shape, np.uint8)
    meta = {"per_frame": []}
    for fi in range(n):
        res = per_frame.get(fi)
        fmeta = {"prompt_to_obj_ids": {}, "object_ids": [], "scores": [], "boxes": []}
        if res is not None and res.get("masks") is not None and len(res["masks"]):
            m = res["masks"]
            m = m.cpu().numpy() if hasattr(m, "cpu") else np.asarray(m)
            if m.ndim == 4:
                m = m[:, 0]
            obj_ids = res["object_ids"]
            obj_ids = obj_ids.tolist() if hasattr(obj_ids, "tolist") else list(obj_ids)
            scores = res["scores"].float().cpu().numpy().tolist()
            boxes = res["boxes"].float().cpu().numpy().tolist()
            p2o = {k: list(v) for k, v in res["prompt_to_obj_ids"].items()}
            obj_to_prompt = {o: p for p, os_ in p2o.items() for o in os_}
            order = np.argsort(scores)  # high score drawn last, wins overlap
            for k in order:
                mask_k = m[k].astype(bool)
                prompt = obj_to_prompt.get(obj_ids[k])
                if prompt is not None:
                    idmasks[prompt][fi][mask_k] = int(obj_ids[k]) + 1
                union[fi][mask_k] = 1
            fmeta = {"prompt_to_obj_ids": p2o, "object_ids": [int(o) for o in obj_ids], "scores": scores, "boxes": boxes}
        meta["per_frame"].append(fmeta)

    for p in args.prompts:
        cov = (idmasks[p].reshape(n, -1).any(1)).sum()
        print(f"  {p!r}: frames with mask {cov}/{n}")

    cutouts, combos, overlays = [], [], []
    for fi in range(n):
        cut = np.empty_like(frames[fi])
        cut[:] = bg_color
        keep = union[fi].astype(bool)
        cut[keep] = frames[fi][keep]
        cutouts.append(cut)
        # overlay colored by object id for visual check
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
        overlays.append(overlay(frames[fi], masks_l, ids_l))
        combos.append(np.concatenate([frames[fi], overlays[-1], cut], axis=1))
    write_video(os.path.join(args.out, "cutout_union.mp4"), cutouts, fps)
    write_video(os.path.join(args.out, "orig_overlay_cutout.mp4"), combos, fps)

    np.savez_compressed(
        os.path.join(args.out, "masks.npz"),
        union=union,
        **{f"idmask_{i}": idmasks[p] for i, p in enumerate(args.prompts)},
    )
    meta.update({
        "video": args.video,
        "prompts": list(args.prompts),
        "idmask_key_to_prompt": {f"idmask_{i}": p for i, p in enumerate(args.prompts)},
        "n_frames": n,
        "idmask_convention": "0=background, k=global object_id k-1",
    })
    with open(os.path.join(args.out, "masks_meta.json"), "w") as f:
        json.dump(meta, f)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
