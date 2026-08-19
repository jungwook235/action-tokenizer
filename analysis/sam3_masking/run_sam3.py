"""SAM3 text-prompted masking on a GR-1 unified episode video.

Two modes:
  - image: per-frame independent Sam3Model inference (no tracking)
  - video: Sam3VideoModel session with text prompt propagated through the video

For each (mode, prompt) an overlay mp4 is written, plus per-prompt
side-by-side comparison videos and summary stats.
"""

import argparse
import json
import os
import time

import cv2
import numpy as np
import torch
from transformers import Sam3Model, Sam3Processor, Sam3VideoModel, Sam3VideoProcessor

PALETTE = np.array(
    [
        [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200],
        [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230],
        [210, 245, 60], [250, 190, 190], [0, 128, 128], [230, 190, 255],
    ],
    dtype=np.uint8,
)


DEFAULT_FPS = 20.0


def load_video(path):
    """Decode every frame; return (frames_rgb, fps).

    fps comes from the container. It used to be hardcoded to 20.0, which happened
    to match gr1_unified and so went unnoticed until ActionNet (15 fps) stamped
    20.0 into its mask npz. fps never reaches SAM3 — it only labels the written
    mp4s and the npz metadata — but a wrong value there is a real bug for anything
    that derives timing from it, so read it instead of assuming.
    """
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not fps or not np.isfinite(fps) or fps <= 0:  # unreadable header
        fps = DEFAULT_FPS
    return frames, float(fps)


def overlay(frame_rgb, masks, ids, scores=None, alpha=0.55):
    """masks: list of HxW bool arrays, ids: color index per mask."""
    out = frame_rgb.astype(np.float32)
    for m, i in zip(masks, ids):
        color = PALETTE[i % len(PALETTE)].astype(np.float32)
        out[m] = out[m] * (1 - alpha) + color * alpha
    out = out.astype(np.uint8)
    # draw contours + score text
    for m, i in zip(masks, ids):
        color = tuple(int(c) for c in PALETTE[i % len(PALETTE)])
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, 1)
    if scores is not None and len(scores):
        txt = " ".join(f"{s:.2f}" for s in scores[:4])
        cv2.putText(out, txt, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def write_video(path, frames_rgb, fps):
    h, w = frames_rgb[0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames_rgb:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()
    # re-encode to h264 for browser playback if ffmpeg available
    tmp = path + ".h264.mp4"
    if os.system(f"ffmpeg -y -v error -i {path} -c:v libx264 -pix_fmt yuv420p {tmp}") == 0:
        os.replace(tmp, path)


def run_image_mode(model, processor, frames, prompt, device, threshold, batch_size=8):
    all_masks, all_scores = [], []
    t0 = time.time()
    for s in range(0, len(frames), batch_size):
        batch = frames[s : s + batch_size]
        inputs = processor(images=batch, text=[prompt] * len(batch), return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model(**inputs)
        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=0.5,
            target_sizes=inputs["original_sizes"].tolist(),
        )
        for r in results:
            masks = r["masks"].cpu().numpy().astype(bool) if len(r["masks"]) else np.zeros((0,) + frames[0].shape[:2], bool)
            scores = r["scores"].float().cpu().numpy() if len(r["masks"]) else np.array([])
            all_masks.append(masks)
            all_scores.append(scores)
    dt = time.time() - t0
    return all_masks, all_scores, dt


def run_video_mode(model, processor, frames, prompt, device, dtype):
    t0 = time.time()
    session = processor.init_video_session(
        video=frames,
        inference_device=device,
        video_storage_device="cpu",
        dtype=dtype,
    )
    processor.add_text_prompt(session, text=prompt)
    per_frame = {}
    with torch.inference_mode():
        for out in model.propagate_in_video_iterator(session, show_progress_bar=True):
            res = processor.postprocess_outputs(session, out)
            per_frame[out.frame_idx] = res
    dt = time.time() - t0
    return per_frame, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--prompts", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True, help="short filename label per prompt")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--model", default="jetjodh/sam3")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--modes", nargs="+", default=["image", "video"])
    args = ap.parse_args()
    assert len(args.prompts) == len(args.labels)

    os.makedirs(args.out, exist_ok=True)
    device = "cuda"
    dtype = torch.bfloat16

    frames, fps = load_video(args.video)
    print(f"video: {args.video} ({len(frames)} frames)")

    stats = {"video": args.video, "n_frames": len(frames), "model": args.model, "runs": []}
    results_store = {}  # (mode, label) -> list per frame of (masks, ids, scores)

    if "image" in args.modes:
        print("loading Sam3Model (image)...")
        img_model = Sam3Model.from_pretrained(args.model, dtype=dtype).to(device).eval()
        img_processor = Sam3Processor.from_pretrained(args.model)
        for prompt, label in zip(args.prompts, args.labels):
            print(f"[image] prompt={prompt!r}")
            masks_l, scores_l, dt = run_image_mode(img_model, img_processor, frames, prompt, device, args.threshold)
            per_frame = []
            for masks, scores in zip(masks_l, scores_l):
                ids = list(range(len(masks)))  # no tracking: color by rank
                per_frame.append((list(masks), ids, list(scores)))
            results_store[("image", label)] = per_frame
            n_inst = [len(m[0]) for m in per_frame]
            stats["runs"].append({
                "mode": "image", "prompt": prompt, "seconds": round(dt, 1),
                "mean_instances": float(np.mean(n_inst)), "frames_with_zero": int(np.sum(np.array(n_inst) == 0)),
                "mean_score": float(np.mean([s for _, _, sc in per_frame for s in sc])) if any(sc for _, _, sc in per_frame) else 0.0,
            })
            print("  ", stats["runs"][-1])
        del img_model
        torch.cuda.empty_cache()

    if "video" in args.modes:
        print("loading Sam3VideoModel...")
        vid_model = Sam3VideoModel.from_pretrained(args.model, dtype=dtype).to(device).eval()
        vid_processor = Sam3VideoProcessor.from_pretrained(args.model)
        for prompt, label in zip(args.prompts, args.labels):
            print(f"[video] prompt={prompt!r}")
            per_frame_raw, dt = run_video_mode(vid_model, vid_processor, frames, prompt, device, dtype)
            per_frame = []
            for fi in range(len(frames)):
                res = per_frame_raw.get(fi)
                masks, ids, scores = [], [], []
                if res is not None:
                    obj_ids = res.get("object_ids", [])
                    obj_ids = obj_ids.tolist() if hasattr(obj_ids, "tolist") else list(obj_ids)
                    m = res.get("masks")
                    sc = res.get("scores", None)
                    if m is not None and len(m):
                        m = m.cpu().numpy() if hasattr(m, "cpu") else np.asarray(m)
                        if m.ndim == 4:
                            m = m[:, 0]
                        masks = [mm.astype(bool) for mm in m]
                        ids = obj_ids if len(obj_ids) == len(masks) else list(range(len(masks)))
                        if sc is not None:
                            sc = sc.float().cpu().numpy() if hasattr(sc, "cpu") else np.asarray(sc)
                            scores = list(sc)
                per_frame.append((masks, ids, scores))
            results_store[("video", label)] = per_frame
            n_inst = [len(m[0]) for m in per_frame]
            stats["runs"].append({
                "mode": "video", "prompt": prompt, "seconds": round(dt, 1),
                "mean_instances": float(np.mean(n_inst)), "frames_with_zero": int(np.sum(np.array(n_inst) == 0)),
                "mean_score": float(np.mean([s for _, _, sc in per_frame for s in sc])) if any(sc for _, _, sc in per_frame) else 0.0,
            })
            print("  ", stats["runs"][-1])
        del vid_model
        torch.cuda.empty_cache()

    # render overlays
    for (mode, label), per_frame in results_store.items():
        rendered = [
            overlay(frames[i], per_frame[i][0], per_frame[i][1], per_frame[i][2])
            for i in range(len(frames))
        ]
        path = os.path.join(args.out, f"{mode}_{label}.mp4")
        write_video(path, rendered, fps)
        print("wrote", path)
        # sample frames
        for fi in [0, len(frames) // 2, len(frames) - 1]:
            cv2.imwrite(
                os.path.join(args.out, f"{mode}_{label}_f{fi:04d}.png"),
                cv2.cvtColor(rendered[fi], cv2.COLOR_RGB2BGR),
            )

    # side-by-side image vs video per prompt
    if "image" in args.modes and "video" in args.modes:
        for label in args.labels:
            im = results_store[("image", label)]
            vd = results_store[("video", label)]
            combo = []
            for i in range(len(frames)):
                a = overlay(frames[i], im[i][0], im[i][1], im[i][2])
                b = overlay(frames[i], vd[i][0], vd[i][1], vd[i][2])
                cv2.putText(a, "image", (4, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(b, "video", (4, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                combo.append(np.concatenate([a, b], axis=1))
            path = os.path.join(args.out, f"compare_{label}.mp4")
            write_video(path, combo, fps)
            print("wrote", path)

    with open(os.path.join(args.out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats["runs"], indent=2))


if __name__ == "__main__":
    main()
