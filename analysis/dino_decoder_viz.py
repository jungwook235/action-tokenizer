"""Visualize a V4 tokenizer's DINO decoder output (future-feature prediction).

The V4 DINO decoder predicts the FUTURE-frame DINO patch features from the
current-frame features + the action latent. For a few validation images per task
we show, side by side:

  cols: [x0 image] [x1 image] [PCA x0] [PCA x1(GT)] [PCA pred] [cos x0] [cos x1] [cos pred] [attn]

  • PCA (DINOv2-style, 2-stage): fit PCA on the patch tokens, use PC1 to split
    foreground/background (Otsu, minority = foreground), then a 2nd PCA on the
    foreground patches → top-3 comps mapped to RGB (background = white). The basis
    is shared across {x0, GT-future x1, predicted-future} of each image so the three
    RGB maps are directly comparable.
  • Cosine map: pick one (foreground, near-center) query patch and show its cosine
    similarity to every other patch — same query index across x0 / x1 / pred.
  • Attn: in the decoder the action-latent tokens are concatenated BEFORE the DINO
    patch tokens; we read the self-attention and show, per patch, how much it
    attends to the action-latent token block (averaged over heads & decoder layers),
    overlaid on the current image.

The wrapper drops the (training-only) dino_decoder, so we rebuild the FULL
ActionLatentTokenizerV4 here and load every key. Self-contained in analysis/.

Run from the action_tokenizer repo root, gr00t-actlat env, on a GPU.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gr00t.experiment.data_config_v3  # noqa: F401
from gr00t.data.dataset_action_frames_v4 import (  # noqa: E402
    ActionFramesCollatorV4, ActionFramesDatasetV4,
)
from gr00t.data.merge_norm_stats import apply_merged_normalization_metadata  # noqa: E402
from gr00t.model.action_latent_tokenizer_v4 import (  # noqa: E402
    ActionLatentTokenizerV4, ReconDecoderV4, TimeWiseEncoderV4,
)
from gr00t.model.rla_modules import SimpleTokenTransformer  # noqa: E402
from gr00t.utils.dino import DINOv3FeatureExtractor  # noqa: E402
from analyze_latents import _encode_mu_and_sample  # noqa: E402

from sklearn.decomposition import PCA  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402


# ---------------------------------------------------------------- model load
def _depth(sd, prefix):
    d = 0
    for k in sd:
        if k.startswith(prefix):
            d = max(d, int(k[len(prefix):].split(".")[0]) + 1)
    return d


def load_full_v4(checkpoint, device):
    """Rebuild the FULL ActionLatentTokenizerV4 (incl. dino_decoder) and load all keys."""
    from safetensors.torch import load_file
    sd = load_file(str(Path(checkpoint) / "model.safetensors"), device="cpu")
    assert "_is_v4" in sd, "not a V4 checkpoint"

    emb_dim, action_dim = sd["encoder.action_encoder.action_proj.weight"].shape
    action_horizon = sd["encoder.action_encoder.time_pos_emb.posembs"].shape[2]
    fusion_width, dino_dim = sd["encoder.joint.input_layer.weight"].shape
    token_dim = sd["encoder.joint.out_layer.weight"].shape[0]
    enc_depth = _depth(sd, "encoder.action_encoder.transformer.blocks.")
    fusion_depth = _depth(sd, "encoder.joint.blocks.")
    dec_depth = _depth(sd, "recon_decoder.transformer.blocks.")
    dino_dec_depth = _depth(sd, "dino_decoder.blocks.")
    fusion_heads = max(1, fusion_width // 64)
    use_vae = "_is_vae" in sd
    dino_final_norm = "naive" if "_dino_final_norm" in sd else "affine"

    print(f"[load] action_dim={action_dim} T={action_horizon} emb={emb_dim} token_dim={token_dim} "
          f"dino_dim={dino_dim} fusion(w{fusion_width},d{fusion_depth},h{fusion_heads}) "
          f"enc{enc_depth} dec{dec_depth} dino_dec{dino_dec_depth} vae={use_vae} norm={dino_final_norm}")

    encoder = TimeWiseEncoderV4(
        action_dim=action_dim, action_horizon=action_horizon, emb_dim=emb_dim, head_dim=64,
        encoder_depth=enc_depth, pdropout=0.0, num_global_tokens=0, num_hand_tokens=0,
        dino_dim=dino_dim, fusion_width=fusion_width, fusion_depth=fusion_depth,
        fusion_heads=fusion_heads, token_dim=token_dim, use_vae=use_vae)
    recon_decoder = ReconDecoderV4(
        action_dim=action_dim, action_horizon=action_horizon, emb_dim=emb_dim, head_dim=64,
        depth=dec_depth, pdropout=0.0, decoder_mode="self_attention",
        num_global_tokens=0, num_hand_tokens=0, token_dim=token_dim)
    dino_decoder = SimpleTokenTransformer(
        in_channels=dino_dim, model_channels=fusion_width, out_channels=dino_dim,
        num_blocks=dino_dec_depth, num_heads=fusion_heads, num_tokens=action_horizon,
        token_channels=token_dim, zero_init=True, use_fp16=False)
    tok = ActionLatentTokenizerV4(
        encoder=encoder, recon_decoder=recon_decoder, dino_decoder=dino_decoder,
        lambda_recon=1.0, lambda_dino=1.0, dino_final_norm=dino_final_norm)
    missing, unexpected = tok.load_state_dict(sd, strict=False)
    missing = [m for m in missing if not m.endswith("_is_v4") and not m.endswith("_is_vae")]
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    tok.eval().to(device)
    for p in tok.parameters():
        p.requires_grad = False
    meta = dict(action_dim=action_dim, T=action_horizon, token_dim=token_dim, dino_dim=dino_dim,
                num_tokens=action_horizon, use_vae=use_vae, dino_final_norm=dino_final_norm)
    return tok, meta


def build_dino(dino_dim, dino_final_norm, device):
    name = {384: "facebook/dinov2-small", 768: "facebook/dinov2-base",
            1024: "facebook/dinov2-large", 1536: "facebook/dinov2-giant"}[dino_dim]
    ex = DINOv3FeatureExtractor(model_name=name, use_compile=False, final_norm=dino_final_norm)
    ex.eval().to(device)
    for p in ex.parameters():
        p.requires_grad = False
    return ex


@torch.no_grad()
def extract_grid(ex, frames_bchw, device):
    f = frames_bchw.to(device).float()
    if f.max() > 1.5:
        f = f / 255.0
    _, grid = ex(f, return_spatial_grid=True)   # [B, C, h, w]
    h, w = grid.shape[-2:]
    feat = grid.flatten(2).transpose(1, 2).float()  # [B, h*w, C]
    return feat, h, w


def build_attn_model(dino_dim, device):
    """Separate eager DINOv2 (sdpa can't return attentions) for CLS→patch attention."""
    from transformers import AutoModel
    name = {384: "facebook/dinov2-small", 768: "facebook/dinov2-base",
            1024: "facebook/dinov2-large", 1536: "facebook/dinov2-giant"}[dino_dim]
    m = AutoModel.from_pretrained(name, attn_implementation="eager").eval().to(device)
    for p in m.parameters():
        p.requires_grad = False
    return m


@torch.no_grad()
def cls_attention(attn_model, ex, frames_bchw, device):
    """Last-layer CLS→patch self-attention (averaged over heads). Returns [B, Lp]."""
    f = frames_bchw.to(device).float()
    if f.max() > 1.5:
        f = f / 255.0
    xpad, _, _ = ex._preprocess(f)
    out = attn_model(xpad, output_attentions=True, return_dict=True)
    attn = out.attentions[-1]                     # [B, heads, seq, seq]
    hH, hW = xpad.shape[2] // ex.patch_size, xpad.shape[3] // ex.patch_size
    num_patches = hH * hW
    num_extra = attn.shape[-1] - num_patches      # CLS (+ any register tokens)
    cls = attn[:, :, 0, num_extra:].mean(1)       # [B, num_patches]
    return cls.float().cpu().numpy(), hH, hW


def kmeans_seg(feats3, k, h, w):
    """Shared KMeans over {x0, x1, pred} patches → 3 label grids (consistent colors)."""
    Lp = h * w
    F = np.concatenate(feats3, 0)
    lab = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(F)
    return [lab[i * Lp:(i + 1) * Lp].reshape(h, w) for i in range(3)]


# ---------------------------------------------------------------- attention
class AttnCapture:
    """Force the dino_decoder's MultiheadAttention layers to return weights."""
    def __init__(self, dino_decoder):
        self.dino_decoder = dino_decoder
        self.store = []
        self._orig = []

    def __enter__(self):
        for blk in self.dino_decoder.blocks:
            mha = blk.attn
            orig = mha.forward
            self._orig.append((mha, orig))

            def wrapped(q, k, v, _orig=orig, **kw):
                kw["need_weights"] = True
                kw["average_attn_weights"] = True
                out, w = _orig(q, k, v, **kw)
                self.store.append(w.detach().float().cpu())   # [B, L, L]
                return out, w
            mha.forward = wrapped
        return self

    def __exit__(self, *a):
        for mha, orig in self._orig:
            mha.forward = orig


# ---------------------------------------------------------------- viz helpers
def _otsu(x):
    hist, edges = np.histogram(x, bins=64, range=(0, 1))
    hist = hist.astype(np.float64)
    tot = hist.sum()
    if tot == 0:
        return 0.5
    w = np.cumsum(hist); wb = w / tot
    centers = (edges[:-1] + edges[1:]) / 2
    csum = np.cumsum(hist * centers)
    mtot = csum[-1] / tot
    best_t, best_v = 0.5, -1
    for i in range(1, 64):
        wB = wb[i]
        wF = 1 - wB
        if wB == 0 or wF == 0:
            continue
        mB = csum[i] / w[i]
        mF = (csum[-1] - csum[i]) / (tot - w[i])
        v = wB * wF * (mB - mF) ** 2
        if v > best_v:
            best_v, best_t = v, centers[i]
    return best_t


def pca_rgb_triplet(feats3, h, w):
    """feats3 = [x0, x1, pred] each [Lp,C]. Shared 2-stage PCA → 3 RGB maps + x0 fg mask."""
    Lp = h * w
    F = np.concatenate(feats3, 0)
    proj = PCA(n_components=3, random_state=0).fit_transform(F)
    c0 = proj[:, 0]
    cn = (c0 - c0.min()) / (np.ptp(c0) + 1e-8)
    thr = _otsu(cn)
    a = cn > thr
    fg = a if a.sum() <= (~a).sum() else ~a   # foreground = minority side
    if fg.sum() < 10:
        fg = np.ones_like(fg)
    p2 = PCA(n_components=3, random_state=0).fit(F[fg])
    rgb = p2.transform(F)
    lo, hi = rgb[fg].min(0), rgb[fg].max(0)
    rgbn = np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1)
    rgbn[~fg] = 1.0
    maps = [rgbn[i * Lp:(i + 1) * Lp].reshape(h, w, 3) for i in range(3)]
    return maps, fg[:Lp]


def pca_rgb_single(feats3, h, w):
    """Single-stage PCA: top-3 comps of the patch features → RGB directly, with NO
    foreground/background split (avoids the PC1-as-background heuristic flipping).
    Shared basis over {x0, x1, pred}; each channel min-maxed over ALL patches."""
    Lp = h * w
    F = np.concatenate(feats3, 0)
    rgb = PCA(n_components=3, random_state=0).fit_transform(F)
    lo, hi = rgb.min(0), rgb.max(0)
    rgbn = np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1)
    return [rgbn[i * Lp:(i + 1) * Lp].reshape(h, w, 3) for i in range(3)]


def cos_map(feat, q, h, w):
    fn = feat / (np.linalg.norm(feat, axis=-1, keepdims=True) + 1e-8)
    return (fn @ fn[q]).reshape(h, w)


def pick_query(fg_x0, h, w):
    cy, cx = h // 2, w // 2
    idx = np.where(fg_x0)[0]
    if len(idx) == 0:
        return cy * w + cx
    rc = np.array([(i // w, i % w) for i in idx])
    d = (rc[:, 0] - cy) ** 2 + (rc[:, 1] - cx) ** 2
    return int(idx[d.argmin()])


def up(a, factor):
    return np.repeat(np.repeat(a, factor, 0), factor, 1)


# ---------------------------------------------------------------- main
def process_task(tok, ex, attn_model, dataset, idxs, device, meta, need_cls=True):
    coll = ActionFramesCollatorV4()
    feats = [dataset[i] for i in idxs]
    batch = coll(feats)
    actions = batch["action"].to(device)
    f0, h, w = extract_grid(ex, batch["frame_x0"], device)
    f1, _, _ = extract_grid(ex, batch["frame_x1"], device)
    mu, sigma, logvar, z = _encode_mu_and_sample(tok.encoder, actions, f0, f1)
    with AttnCapture(tok.dino_decoder) as cap:
        pred = tok.decode_dino(z.to(tok.encoder.action_proj.weight.dtype), f0)  # [B,Lp,C]
    # decoder attention: weights list len = num_blocks, each [B, L, L]; L = num_tokens + Lp
    nt = meta["num_tokens"]
    W = torch.stack(cap.store, 0)           # [nblk, B, L, L]
    patch_to_latent = W[:, :, nt:, :nt].sum(-1).mean(0)  # [B, Lp]
    # backbone CLS→patch attention on the real frames (only if needed)
    if need_cls:
        cls0, _, _ = cls_attention(attn_model, ex, batch["frame_x0"], device)
        cls1, _, _ = cls_attention(attn_model, ex, batch["frame_x1"], device)
    else:
        cls0 = cls1 = None
    out = dict(
        x0=batch["frame_x0"].permute(0, 2, 3, 1).numpy(),   # [B,H,W,3] uint8
        x1=batch["frame_x1"].permute(0, 2, 3, 1).numpy(),
        f0=f0.cpu().numpy(), f1=f1.cpu().numpy(), pred=pred.cpu().numpy(),
        attn=patch_to_latent.numpy(), cls0=cls0, cls1=cls1, h=h, w=w,
    )
    return out


def make_task_figure(out, tag, task_name, png):
    B, h, w = len(out["x0"]), out["h"], out["w"]
    up_f = 224 // h if h > 0 else 14
    cols = ["x0 (current)", "x1 (GT future)", "PCA x0", "PCA x1(GT)", "PCA pred",
            "cos x0", "cos x1(GT)", "cos pred", "attn→latent (x0)"]
    fig, axes = plt.subplots(B, 9, figsize=(9 * 1.9, B * 1.9), squeeze=False)
    for r in range(B):
        x0img, x1img = out["x0"][r], out["x1"][r]
        maps, fg0 = pca_rgb_triplet([out["f0"][r], out["f1"][r], out["pred"][r]], h, w)
        q = pick_query(fg0, h, w)
        cos = [cos_map(out["f0"][r], q, h, w), cos_map(out["f1"][r], q, h, w),
               cos_map(out["pred"][r], q, h, w)]
        qy, qx = q // w, q % w
        attn = out["attn"][r].reshape(h, w)
        attn = (attn - attn.min()) / (np.ptp(attn) + 1e-8)

        panels = [
            ("img", x0img), ("img", x1img),
            ("rgb", maps[0]), ("rgb", maps[1]), ("rgb", maps[2]),
            ("cos", cos[0]), ("cos", cos[1]), ("cos", cos[2]),
            ("attn", (x0img, attn)),
        ]
        for c, (kind, data) in enumerate(panels):
            ax = axes[r][c]
            if kind == "img":
                ax.imshow(data)
                if c == 0:  # mark query location on the current image
                    ax.scatter([qx * up_f + up_f / 2], [qy * up_f + up_f / 2],
                               c="red", s=40, marker="x")
            elif kind == "rgb":
                ax.imshow(data, interpolation="bilinear")
            elif kind == "cos":
                ax.imshow(data, cmap="turbo", vmin=float(data.min()), vmax=1.0,
                          interpolation="bilinear")
            else:
                img, heat = data
                ax.imshow(img)
                ax.imshow(up(heat, up_f), cmap="turbo", alpha=0.5)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[c], fontsize=9)
    fig.suptitle(f"{tag}  —  DINO-decoder future-feature viz  —  task: {task_name}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(png, dpi=125)
    plt.close(fig)


def make_pca_attn_figure(out, tag, task_name, png):
    """Minimal figure: single-stage PCA-RGB (no fg/bg split) + decoder attn→latent."""
    B, h, w = len(out["x0"]), out["h"], out["w"]
    up_f = 224 // h if h > 0 else 14
    cols = ["x0 (current)", "x1 (GT future)", "PCA x0", "PCA x1(GT)", "PCA pred",
            "attn→latent (x0)"]
    fig, axes = plt.subplots(B, 6, figsize=(6 * 1.9, B * 1.9), squeeze=False)
    for r in range(B):
        x0img, x1img = out["x0"][r], out["x1"][r]
        maps = pca_rgb_single([out["f0"][r], out["f1"][r], out["pred"][r]], h, w)
        attn = out["attn"][r].reshape(h, w)
        attn = (attn - attn.min()) / (np.ptp(attn) + 1e-8)
        panels = [("img", x0img), ("img", x1img),
                  ("rgb", maps[0]), ("rgb", maps[1]), ("rgb", maps[2]),
                  ("attn", (x0img, attn))]
        for c, (kind, data) in enumerate(panels):
            ax = axes[r][c]
            if kind == "img":
                ax.imshow(data)
            elif kind == "rgb":
                ax.imshow(data, interpolation="bilinear")
            else:
                img, heat = data
                ax.imshow(img)
                ax.imshow(up(heat, up_f), cmap="turbo", alpha=0.5)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[c], fontsize=9)
    fig.suptitle(f"{tag}  —  PCA(no fg/bg split) + attn→latent  —  task: {task_name}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(png, dpi=125)
    plt.close(fig)


def make_attn_kmeans_figure(out, kmeans_k, tag, task_name, png):
    B, h, w = len(out["x0"]), out["h"], out["w"]
    up_f = 224 // h if h > 0 else 14
    cols = ["x0 (current)", "CLS-attn x0", "kmeans x0",
            "x1 (GT future)", "CLS-attn x1", "kmeans x1(GT)", "kmeans pred"]
    fig, axes = plt.subplots(B, 7, figsize=(7 * 1.9, B * 1.9), squeeze=False)
    for r in range(B):
        x0img, x1img = out["x0"][r], out["x1"][r]
        seg = kmeans_seg([out["f0"][r], out["f1"][r], out["pred"][r]], kmeans_k, h, w)
        a0 = out["cls0"][r].reshape(h, w); a0 = (a0 - a0.min()) / (np.ptp(a0) + 1e-8)
        a1 = out["cls1"][r].reshape(h, w); a1 = (a1 - a1.min()) / (np.ptp(a1) + 1e-8)
        panels = [
            ("img", x0img), ("attn", (x0img, a0)), ("seg", seg[0]),
            ("img", x1img), ("attn", (x1img, a1)), ("seg", seg[1]), ("seg", seg[2]),
        ]
        for c, (kind, data) in enumerate(panels):
            ax = axes[r][c]
            if kind == "img":
                ax.imshow(data)
            elif kind == "attn":
                img, heat = data
                ax.imshow(img)
                ax.imshow(up(heat, up_f), cmap="turbo", alpha=0.5)
            else:
                ax.imshow(data, cmap="tab10", vmin=0, vmax=max(9, kmeans_k - 1),
                          interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[c], fontsize=9)
    fig.suptitle(f"{tag}  —  CLS-attention + k-means(k={kmeans_k}) segmentation  —  task: {task_name}",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(png, dpi=125)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-config", required=True)
    ap.add_argument("--dataset-path", nargs="+", required=True)
    ap.add_argument("--embodiment-tag", default="new_embodiment")
    ap.add_argument("--normalization-mode", default="min_max")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--video-backend", default="decord")
    ap.add_argument("--val-ratio", type=float, default=0.003)
    ap.add_argument("--val-seed", type=int, default=42)
    ap.add_argument("--fixed-val-path", default=None)
    ap.add_argument("--images-per-task", type=int, default=5)
    ap.add_argument("--tasks-max", type=int, default=6)
    ap.add_argument("--kmeans-k", type=int, default=6)
    ap.add_argument("--figures", nargs="+", default=["dino_viz", "attn_kmeans", "pca_attn"],
                    choices=["dino_viz", "attn_kmeans", "pca_attn"],
                    help="which figures to generate per task")
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    figset = set(args.figures)

    outdir = _REPO_ROOT / "analysis" / "output" / "dino_viz" / args.tag
    outdir.mkdir(parents=True, exist_ok=True)

    tok, meta = load_full_v4(args.checkpoint, device)
    ex = build_dino(meta["dino_dim"], meta["dino_final_norm"], device)
    need_cls = "attn_kmeans" in figset
    attn_model = build_attn_model(meta["dino_dim"], device) if need_cls else None

    paths = args.dataset_path[: args.tasks_max]
    rng = np.random.default_rng(args.sample_seed)
    for p in paths:
        name = Path(p).name
        ds = ActionFramesDatasetV4(
            dataset_path=p, data_config_name=args.data_config, embodiment_tag=args.embodiment_tag,
            split="val", val_ratio=args.val_ratio, val_seed=args.val_seed,
            normalization_mode=args.normalization_mode, image_size=args.image_size,
            video_backend=args.video_backend, use_fixed_val=True, fixed_val_path=args.fixed_val_path)
        apply_merged_normalization_metadata([ds], [ds])
        n = len(ds)
        k = min(args.images_per_task, n)
        idxs = sorted(rng.choice(n, size=k, replace=False).tolist())
        print(f"[viz] task={name}: {n} val chunks → {k} images")
        out = process_task(tok, ex, attn_model, ds, idxs, device, meta, need_cls=need_cls)
        if "dino_viz" in figset:
            p = outdir / f"{name[:60]}_dino_viz.png"
            make_task_figure(out, args.tag, name[:48], p); print(f"[viz] wrote {p}")
        if "attn_kmeans" in figset:
            p = outdir / f"{name[:60]}_attn_kmeans.png"
            make_attn_kmeans_figure(out, args.kmeans_k, args.tag, name[:48], p); print(f"[viz] wrote {p}")
        if "pca_attn" in figset:
            p = outdir / f"{name[:60]}_pca_attn.png"
            make_pca_attn_figure(out, args.tag, name[:48], p); print(f"[viz] wrote {p}")

    print(f"[viz] done → {outdir}")


if __name__ == "__main__":
    main()
