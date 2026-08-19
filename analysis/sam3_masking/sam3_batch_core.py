"""Shared machinery for the role-separated SAM3 batch masking scripts.

Front-ends (batch_sam3_actionnet.py, batch_sam3_gr1_unified.py) only decide how to
enumerate episodes and which prompts each one gets; everything below — the single
multi-prompt video session, the cutout/overlay rendering, and the npz layout that
keeps the robot masks separable from the object masks — lives here.

npz layout written by write_outputs (uint8 0/1, frame t == step t of the parquet):
    mask          (T,H,W)    union of every prompt                [legacy-compatible]
    robot_mask    (T,H,W)    union of the robot prompts only
    object_mask   (T,H,W)    union of the task-noun prompts only
    prompt_masks  (P,T,H,W)  one plane per prompt                 [legacy-compatible]
    prompts       (P,)       prompt strings, same order as prompt_masks
    prompt_roles  (P,)       "robot" / "object", parallel to prompts
    n_robot_prompts ()       prompt_masks[:n] is the robot group, [n:] the objects
    episode_index (), fps ()
"""

import numpy as np
import torch

from run_sam3 import PALETTE, write_video

BG_COLORS = {"black": (0, 0, 0), "white": (255, 255, 255), "green": (0, 177, 64)}

# Fixed robot-side prompts. Probed on both corpora: this pair covers both arms from
# frame 0, while "Robot" alone silently drops the idle arm on some episodes.
ROBOT_PROMPTS = ["robot hand", "robot arm"]


def load_model(model_id, device="cuda", dtype=torch.bfloat16):
    from transformers import Sam3VideoModel, Sam3VideoProcessor
    model = Sam3VideoModel.from_pretrained(model_id, dtype=dtype).to(device).eval()
    processor = Sam3VideoProcessor.from_pretrained(model_id)
    return model, processor


def run_session(model, processor, frames, prompts, device, dtype):
    """Propagate all prompts in ONE video session -> (P, T, H, W) uint8 per-prompt masks."""
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
    """Colour each prompt with its own palette entry, so the robot group (prompt
    indices 0..n_robot-1) stays visually distinct from the objects."""
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


def cutout_frames(frames, mask, bg_color):
    out = []
    for fi, fr in enumerate(frames):
        cut = np.empty_like(fr)
        cut[:] = bg_color
        keep = mask[fi].astype(bool)
        cut[keep] = fr[keep]
        out.append(cut)
    return out


def split_masks(prompt_masks, n_robot):
    """(robot_mask, object_mask, union) as (T,H,W) uint8."""
    robot = prompt_masks[:n_robot].any(0).astype(np.uint8)
    object_ = (prompt_masks[n_robot:].any(0).astype(np.uint8)
               if prompt_masks.shape[0] > n_robot else np.zeros_like(robot))
    return robot, object_, prompt_masks.any(0).astype(np.uint8)


def per_prompt_stats(prompt_masks, prompts):
    st = {}
    n = prompt_masks.shape[1]
    for pi, p in enumerate(prompts):
        pm = prompt_masks[pi]
        cov = pm.reshape(n, -1).any(1)
        det = int(cov.sum())
        st[p] = {"frames_detected": det,
                 "mean_area_pct": round(float(pm[cov].mean() * 100), 2) if det else 0.0}
    return st


def write_outputs(frames, fps, prompt_masks, prompts, n_robot, ep,
                  cut_path, ovl_path, msk_path, bg_color,
                  cutout_from="union", keep_prompt_masks=True, write_videos=True,
                  write_overlay=True):
    """Render the videos (unless write_videos=False) and save the role-separated npz.

    ``write_overlay=False`` writes the cutout but skips the overlay. The overlay exists
    for humans to spot-check the masks -- nothing downstream reads it -- so a caller
    processing tens of thousands of episodes should render only a sample of them.

    Returns the stats dict that the caller logs to the manifest.
    """
    robot_mask, object_mask, union = split_masks(prompt_masks, n_robot)
    if write_videos:
        source = {"union": union, "robot": robot_mask, "object": object_mask}[cutout_from]
        write_video(cut_path, cutout_frames(frames, source, bg_color), fps)
        if write_overlay:
            write_video(ovl_path, overlay_by_prompt(frames, prompt_masks), fps)

    payload = dict(
        mask=union,
        robot_mask=robot_mask,
        object_mask=object_mask,
        prompts=np.array(prompts),
        prompt_roles=np.array(["robot"] * n_robot + ["object"] * (len(prompts) - n_robot)),
        n_robot_prompts=n_robot,
        episode_index=ep,
        fps=fps,
    )
    if keep_prompt_masks:
        payload["prompt_masks"] = prompt_masks
    np.savez_compressed(msk_path, **payload)

    n = len(frames)
    return {
        "n_frames": n, "fps": fps, "prompts": prompts, "n_robot_prompts": n_robot,
        "frames_with_mask": int(union.reshape(n, -1).any(1).sum()),
        "robot_frames_with_mask": int(robot_mask.reshape(n, -1).any(1).sum()),
        "object_frames_with_mask": int(object_mask.reshape(n, -1).any(1).sum()),
        "per_prompt": per_prompt_stats(prompt_masks, prompts),
    }
