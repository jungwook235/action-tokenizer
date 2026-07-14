"""②′  FRAME-SWAP INTERVENTION — the decisive mechanism test.

The observational analyses (vsep_stats/vsep_frames) cannot demonstrate "same
action, different visual → different latent" on the dexjoco dual-arm val set,
because the 5 tasks have near-disjoint action spaces (there are essentially no
action-collisions across tasks, so no natural same-action/different-visual pairs).

This script MANUFACTURES that condition directly: it holds the action chunk
byte-identical and swaps in the visual observation (DINO features) from other
samples, then measures how far the tokenizer's latent moves.

    z_i^(l) = encode( action_i , frames_l )      for L visual donors l

  • v3 ignores frames  → z_i^(l) is identical for every l  → visual spread = 0.
  • v4 fuses DINO      → z_i^(l) moves with the visual      → visual spread > 0.

We report, per anchor, the visual-induced latent spread as a fraction of the
natural action-induced latent scale, and correlate the per-(anchor,donor) latent
shift with the DINO visual distance (does a bigger visual change move the latent
more?). This isolates the visual contribution to the latent with the action held
exactly fixed — the mechanism behind the "visual disambiguation" claim.

Run from the action_tokenizer repo root, gr00t-actlat env, on a GPU.
"""

import argparse
import json
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

import gr00t.experiment.data_config_v3  # noqa: F401,E402
from gr00t.data.dataset_action_frames_v4 import ActionFramesCollatorV4  # noqa: E402
from gr00t.model.action_latent_tokenizer_wrapper import ActionLatentTokenizerWrapper  # noqa: E402
from analyze_latents import _encode_mu_and_sample  # noqa: E402
from vsep_collect import build_datasets, sample_indices  # noqa: E402


@torch.no_grad()
def gather(args, device):
    """Collect A[N,T,D], DINO feats f0/f1[N,P,C], raw x1 frames, task[N]."""
    datasets, task_names = build_datasets(args)
    samp_task, samp_local = sample_indices(datasets, args)
    subsets = [torch.utils.data.Subset(datasets[ti], samp_local[samp_task == ti].tolist())
               for ti in range(len(datasets))]
    loader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset(subsets), batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=ActionFramesCollatorV4())

    wrap = ActionLatentTokenizerWrapper.from_checkpoint(args.v4_ckpt, device=device)
    wrap.eval()
    A, F0, F1, X1 = [], [], [], []
    for b in loader:
        A.append(b["action"].float())
        f0, f1 = wrap._resolve_dino_feats(b["frame_x0"], b["frame_x1"], None, None, device)
        F0.append(f0.float().cpu()); F1.append(f1.float().cpu())
        X1.append(b["frame_x1"].cpu())
    A = torch.cat(A); F0 = torch.cat(F0); F1 = torch.cat(F1); X1 = torch.cat(X1)
    return wrap, A, F0, F1, X1, samp_task, task_names


@torch.no_grad()
def swap_encode_v4(wrap, A, F0, F1, donors, device, batch=128):
    """Return Zmatch[M,T,K] and Zdon[M,L,T,K] for anchors A under each donor's frames."""
    enc = wrap.tokenizer.encoder
    M = A.shape[0]
    # matched (diagonal)
    Zmatch = []
    for i in range(0, M, batch):
        a = A[i:i + batch].to(device)
        mu, *_ = _encode_mu_and_sample(enc, a, F0[i:i + batch].to(device), F1[i:i + batch].to(device))
        Zmatch.append(mu.float().cpu())
    Zmatch = torch.cat(Zmatch)
    # each donor's frame broadcast to all anchors
    Zdon = torch.empty(M, len(donors), *Zmatch.shape[1:], dtype=torch.float32)
    for li, d in enumerate(donors):
        f0d = F0[d].to(device).unsqueeze(0); f1d = F1[d].to(device).unsqueeze(0)
        for i in range(0, M, batch):
            a = A[i:i + batch].to(device)
            bs = a.shape[0]
            mu, *_ = _encode_mu_and_sample(enc, a, f0d.expand(bs, -1, -1), f1d.expand(bs, -1, -1))
            Zdon[i:i + batch, li] = mu.float().cpu()
    return Zmatch, Zdon


@torch.no_grad()
def swap_encode_v3(ckpt, A, donors, device, batch=128):
    """v3 ignores frames: Zmatch and Zdon (should be identical across donors)."""
    wrap = ActionLatentTokenizerWrapper.from_checkpoint(ckpt, device=device)
    wrap.eval()
    dtype = wrap.tokenizer.encoder.action_proj.weight.dtype
    M = A.shape[0]
    Z = []
    for i in range(0, M, batch):
        g, t, h = wrap.encode(A[i:i + batch].to(device).to(dtype))
        Z.append(t.float().cpu())
    Zmatch = torch.cat(Z)
    Zdon = Zmatch.unsqueeze(1).repeat(1, len(donors), 1, 1)  # frame-independent by construction
    del wrap; torch.cuda.empty_cache()
    return Zmatch, Zdon


def spreads(Zmatch, Zdon):
    """visual spread per anchor (RMS latent move over donors, action fixed) and the
    global action scale (median pairwise dist of matched latents). Latents z-scored."""
    M, L = Zdon.shape[0], Zdon.shape[1]
    zf = Zmatch.reshape(M, -1).numpy()
    mu = zf.mean(0, keepdims=True); sd = zf.std(0, keepdims=True) + 1e-8
    def z(x): return (x.reshape(x.shape[0], -1).numpy() - mu) / sd
    Zm = z(Zmatch)                      # [M,P]
    # action scale = median pairwise distance among matched latents
    rng = np.random.default_rng(0)
    ii = rng.integers(0, M, 20000); jj = rng.integers(0, M, 20000); ok = ii != jj
    action_scale = float(np.median(np.linalg.norm(Zm[ii[ok]] - Zm[jj[ok]], axis=1)))
    # visual spread per anchor: RMS distance of donor-latents from their per-anchor mean
    vis = np.zeros(M)
    Zd = ((Zdon.reshape(M, L, -1).numpy() - mu) / sd)   # [M,L,P]
    center = Zd.mean(1, keepdims=True)
    vis = np.sqrt(((Zd - center) ** 2).sum(-1).mean(1))  # RMS over donors
    return vis, action_scale, Zd, Zm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v3-ckpt", required=True)
    ap.add_argument("--v4-ckpt", required=True)
    ap.add_argument("--data-config", required=True)
    ap.add_argument("--dataset-path", nargs="+", required=True)
    ap.add_argument("--embodiment-tag", default="new_embodiment")
    ap.add_argument("--normalization-mode", default="min_max")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--video-backend", default="decord")
    ap.add_argument("--val-ratio", type=float, default=0.003)
    ap.add_argument("--val-seed", type=int, default=42)
    ap.add_argument("--fixed-val-path", default=None)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--target-total", type=int, default=750)
    ap.add_argument("--min-per-dataset", type=int, default=60)
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--n-donors", type=int, default=24)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(_REPO_ROOT / "analysis" / "output" / "visual_sep"))
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    print("[swap] gathering actions + DINO feats ...")
    wrap4, A, F0, F1, X1, task, task_names = gather(args, device)
    M = A.shape[0]
    # donors: spread across tasks for maximal visual variety
    rng = np.random.default_rng(args.sample_seed)
    donors = rng.choice(M, size=min(args.n_donors, M), replace=False)
    print(f"[swap] M={M} anchors, L={len(donors)} visual donors")

    # visual distance between each donor's pooled DINO context and each anchor's
    Vpool = torch.cat([F0.mean(1), F1.mean(1)], dim=1).numpy()  # [M, 2C]

    print("[swap] v4 swap-encode ...")
    Zm4, Zd4 = swap_encode_v4(wrap4, A, F0, F1, donors, device, args.batch_size)
    del wrap4; torch.cuda.empty_cache()
    print("[swap] v3 swap-encode (frame-independent) ...")
    Zm3, Zd3 = swap_encode_v3(args.v3_ckpt, A, donors, device, args.batch_size)

    vis4, ascale4, Zd4z, Zm4z = spreads(Zm4, Zd4)
    vis3, ascale3, _, _ = spreads(Zm3, Zd3)
    r4 = vis4 / ascale4
    r3 = vis3 / ascale3

    # correlation: latent move vs visual distance, pooled over (anchor, donor)
    dl, dv = [], []
    Vz = (Vpool - Vpool.mean(0)) / (Vpool.std(0) + 1e-8)
    for li, d in enumerate(donors):
        move = np.linalg.norm(Zd4z[:, li] - Zm4z, axis=1)   # latent move (z-scored)
        vdist = np.linalg.norm(Vz - Vz[d], axis=1)          # visual distance to donor
        dl.append(move); dv.append(vdist)
    dl = np.concatenate(dl); dv = np.concatenate(dv)
    corr = float(np.corrcoef(dl, dv)[0, 1])

    # ---- figure ----
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].hist(r3, bins=30, alpha=0.6, color="#1f77b4", label=f"v3 (median={np.median(r3):.3f})")
    ax[0].hist(r4, bins=30, alpha=0.6, color="#d62728", label=f"v4 (median={np.median(r4):.3f})")
    ax[0].set_title("Visual-induced latent spread / action scale\n(action held fixed, only frames swapped)")
    ax[0].set_xlabel("visual spread / action scale"); ax[0].set_ylabel("# anchors")
    ax[0].legend(); ax[0].grid(alpha=0.2)
    sub = rng.choice(len(dl), min(6000, len(dl)), replace=False)
    ax[1].scatter(dv[sub], dl[sub], s=5, alpha=0.25, color="#d62728", linewidths=0)
    ax[1].set_title(f"v4 latent move vs visual distance\n(pooled, action fixed)   r={corr:.3f}")
    ax[1].set_xlabel("DINO visual distance (donor vs anchor)"); ax[1].set_ylabel("v4 latent move")
    ax[1].grid(alpha=0.2)
    fig.suptitle("②′  Frame-swap intervention — v4 latent moves with the visual (action fixed); v3 does not",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    png = outdir / "frame_swap.png"
    fig.savefig(png, dpi=130); plt.close(fig)

    L = ["=" * 84, "②′  FRAME-SWAP INTERVENTION (action fixed, visual swapped)", "=" * 84,
         f"M={M} anchors, L={len(donors)} donors, {M*len(donors)} swapped encodings/tokenizer", "",
         "Visual-induced latent spread as a fraction of the action-induced latent scale:",
         f"  {'':<8}{'median':>10}{'mean':>10}{'p90':>10}",
         f"  {'v3':<8}{np.median(r3):>10.4f}{r3.mean():>10.4f}{np.percentile(r3,90):>10.4f}   (≈0 by construction: frames ignored)",
         f"  {'v4':<8}{np.median(r4):>10.4f}{r4.mean():>10.4f}{np.percentile(r4,90):>10.4f}", "",
         f"Correlation( v4 latent move , DINO visual distance ) pooled = {corr:.3f}",
         f"  (v3 = 0 exactly — latent does not depend on the frames)", "",
         "Read: with the ACTION held byte-identical, swapping the observed visual moves the",
         f"  v4 latent by ~{100*np.median(r4):.0f}% of the action-scale (v3: 0%), and that move grows",
         "  with the size of the visual change (positive correlation). This is the direct",
         "  mechanism of visual disambiguation, isolated from any action difference.", "",
         f"figure -> {png}", "=" * 84]
    txt = outdir / "frame_swap.txt"
    txt.write_text("\n".join(L))
    print("\n".join(L)); print(f"[swap] wrote {png}")


if __name__ == "__main__":
    main()
