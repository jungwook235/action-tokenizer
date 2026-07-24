"""DINO-decoder future-feature viz for a hand-extracted single chunk (sk_data).

Same visualization as ``analysis/dino_decoder_viz.py`` (PCA-RGB of {x0, GT-x1,
predicted-future} DINO features + decoder attn->action-latent), but instead of a
LeRobot dataset it reads a self-contained sample directory:

    sk_data/
      ep<E>_action_chunk_<a>_<b>.json   # {"actions": [[...44...] x T], "modality": {...}}
      ep<E>_frame_<a>.png               # current  frame  (x0)
      ep<E>_frame_<b>.png               # future   frame  (x1)

The action JSON carries the FULL 44-dim gr1 action; the V4 gr1 tokenizer consumes
only [left_arm, right_arm, left_hand, right_hand, waist] (=29 dims, that concat
order), min_max-normalized. We reproduce the training normalization by merging
min/max across every gr1_unified.* task (min-of-mins / max-of-maxs, exactly what
``apply_merged_normalization_metadata`` does for the ConcatDataset the checkpoint
was trained on).

Run from the repo root, gr00t-actlat env, on a GPU.
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dino_decoder_viz import (  # noqa: E402
    AttnCapture, build_dino, extract_grid, load_full_v4,
    make_pca_attn_figure, make_task_figure,
)
from analyze_latents import _encode_mu_and_sample  # noqa: E402

# fourier_gr1_arms_waist action keys, in concat order, with their slice into the
# 44-dim raw gr1 action vector (see any gr1_unified meta/modality.json).
ACTION_KEY_SLICES = [
    ("left_arm", 0, 7),
    ("right_arm", 22, 29),
    ("left_hand", 7, 13),
    ("right_hand", 29, 35),
    ("waist", 41, 44),
]
GR1_UNIFIED_GLOB = (
    "/storage1/sjw_dataset/dataset/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/"
    "gr1_unified.*"
)


def merged_minmax(glob_pat, dim=44):
    """min-of-mins / max-of-maxs of the 44-dim 'action' across all matched tasks."""
    paths = sorted(glob.glob(glob_pat))
    assert paths, f"no datasets match {glob_pat}"
    mn = np.full(dim, np.inf)
    mx = np.full(dim, -np.inf)
    for p in paths:
        s = json.load(open(Path(p) / "meta" / "stats.json"))["action"]
        mn = np.minimum(mn, np.asarray(s["min"], dtype=np.float64))
        mx = np.maximum(mx, np.asarray(s["max"], dtype=np.float64))
    print(f"[norm] merged min/max over {len(paths)} gr1_unified tasks")
    return mn, mx


def normalize_action(actions44, mn, mx):
    """actions44: [T, 44] raw -> [T, 29] min_max normalized in ACTION_KEY_SLICES order.

    Formula matches StateActionTransform min_max: 2*(x-min)/(max-min)-1, and 0
    where min==max (constant dims)."""
    cols = []
    for _, a, b in ACTION_KEY_SLICES:
        x = actions44[:, a:b].astype(np.float64)
        lo, hi = mn[a:b], mx[a:b]
        rng = hi - lo
        mask = rng != 0
        out = np.zeros_like(x)
        out[:, mask] = 2 * (x[:, mask] - lo[mask]) / rng[mask] - 1
        cols.append(out)
    return np.concatenate(cols, axis=1).astype(np.float32)  # [T, 29]


def load_frame(png, size=224):
    """PNG -> [H,W,3] uint8, bilinear-resized to (size,size) (matches VideoResize)."""
    img = Image.open(png).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def find_sample(data_dir):
    """Locate the (json, x0_png, x1_png) triple inside data_dir."""
    jsons = sorted(Path(data_dir).glob("*action_chunk*.json"))
    assert jsons, f"no *action_chunk*.json in {data_dir}"
    j = jsons[0]
    meta = json.load(open(j))
    a, b = meta["frame_start"], meta["frame_end"]
    ep = meta["episode"]
    x0 = Path(data_dir) / f"ep{ep}_frame_{a}.png"
    x1 = Path(data_dir) / f"ep{ep}_frame_{b}.png"
    assert x0.exists() and x1.exists(), f"missing frame pngs {x0} / {x1}"
    return j, meta, x0, x1


@torch.no_grad()
def run(tok, ex, meta, actions_norm, frame0, frame1, device):
    """Return the 'out' dict make_*_figure expects, for a single sample (B=1)."""
    actions = torch.from_numpy(actions_norm)[None].to(device)          # [1,T,29]
    f0img = torch.from_numpy(frame0).permute(2, 0, 1)[None]            # [1,3,H,W] uint8
    f1img = torch.from_numpy(frame1).permute(2, 0, 1)[None]
    f0, h, w = extract_grid(ex, f0img, device)
    f1, _, _ = extract_grid(ex, f1img, device)
    _, _, _, z = _encode_mu_and_sample(tok.encoder, actions, f0, f1)
    with AttnCapture(tok.dino_decoder) as cap:
        pred = tok.decode_dino(z.to(tok.encoder.action_proj.weight.dtype), f0)  # [1,Lp,C]
    nt = meta["num_tokens"]
    W = torch.stack(cap.store, 0)                                     # [nblk,1,L,L]
    attn = W[:, :, nt:, :nt].sum(-1).mean(0)                          # [1,Lp]
    return dict(
        x0=f0img.permute(0, 2, 3, 1).numpy(), x1=f1img.permute(0, 2, 3, 1).numpy(),
        f0=f0.cpu().numpy(), f1=f1.cpu().numpy(), pred=pred.cpu().numpy(),
        attn=attn.cpu().numpy(), cls0=None, cls1=None, h=h, w=w,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(_REPO_ROOT / "analysis" / "sk_data"))
    ap.add_argument("--checkpoint", default=(
        "/sjw_alinlab1/home/jungwook/Isaac-GR00T/checkpoints_action_tokenizer/"
        "gr1_1000demos_v4_recon_dino_bn64_l1_mse_naiveln_vae/checkpoint-100000"))
    ap.add_argument("--tag", default="sk_data")
    ap.add_argument("--norm-glob", default=GR1_UNIFIED_GLOB,
                    help="glob of datasets whose min/max are merged for normalization")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--figures", nargs="+", default=["pca_attn", "dino_viz"],
                    choices=["pca_attn", "dino_viz"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    j, smeta, x0png, x1png = find_sample(args.data_dir)
    name = j.stem
    print(f"[skdata] sample={name} ep={smeta['episode']} "
          f"frames {smeta['frame_start']}->{smeta['frame_end']} "
          f"action_dim={smeta['action_dim']} T={smeta['chunk_length']}")

    tok, meta = load_full_v4(args.checkpoint, device)
    assert meta["action_dim"] == 29, f"expected 29-dim gr1 action, got {meta['action_dim']}"
    ex = build_dino(meta["dino_dim"], meta["dino_final_norm"], device)

    actions44 = np.asarray(smeta["actions"], dtype=np.float64)         # [T,44]
    assert actions44.shape[1] == 44, f"expected 44-dim raw action, got {actions44.shape}"
    T = meta["T"]
    if actions44.shape[0] != T:
        print(f"[skdata] chunk len {actions44.shape[0]} != model horizon {T}; "
              f"{'truncating' if actions44.shape[0] > T else 'padding(last)'}")
        if actions44.shape[0] > T:
            actions44 = actions44[:T]
        else:
            pad = np.repeat(actions44[-1:], T - actions44.shape[0], axis=0)
            actions44 = np.concatenate([actions44, pad], axis=0)

    mn, mx = merged_minmax(args.norm_glob)
    actions_norm = normalize_action(actions44, mn, mx)                # [T,29]
    frame0 = load_frame(x0png, args.image_size)
    frame1 = load_frame(x1png, args.image_size)

    out = run(tok, ex, meta, actions_norm, frame0, frame1, device)

    outdir = Path(args.data_dir)
    if "pca_attn" in args.figures:
        p = outdir / f"{name}_pca_attn.png"
        make_pca_attn_figure(out, args.tag, name, p); print(f"[skdata] wrote {p}")
    if "dino_viz" in args.figures:
        p = outdir / f"{name}_dino_viz.png"
        make_task_figure(out, args.tag, name, p); print(f"[skdata] wrote {p}")
    print("[skdata] done")


if __name__ == "__main__":
    main()
