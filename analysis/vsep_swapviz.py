"""②‴  FIXED-ACTION / VARYING-VISUAL — the exact "v3 = one point, v4 = spread" plot.

Real near-dup groups still carry a tiny residual action difference, so v3 does not
fully collapse. To show the intended claim *exactly*, we FIX the action and vary
only the visual:

  1. Find a near-duplicate action group G (real chunks with ~identical action).
  2. Pick the group's medoid action  a*  (one action chunk).
  3. Re-encode with a* held BYTE-IDENTICAL, but swapping in each member's real
     (x0, x1) frames:   z_i = encode(a*, DINO(frame_i)).
       • v3 ignores frames  → every z_i is the SAME point  (spread = 0, exactly).
       • v4 fuses the frames → the z_i SPREAD OUT, organized by the visual.

Point-distribution figure: left = v3 (one marker, "N→1"), right = v4 (colored
cloud), both scaled to global-median latent units with a shared axis; below, the
x0→x1 frame pairs for K members sampled across the visual axis, color-linked to
their v4 points. The motion (x0→x1) looks the same across pairs (same action); the
scene/context differs — and only v4 reflects it.

Run from repo root, gr00t-actlat env, on a GPU.
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

import gr00t.experiment.data_config_v3  # noqa: F401,E402
from gr00t.data.dataset_action_frames_v4 import ActionFramesCollatorV4  # noqa: E402
from gr00t.model.action_latent_tokenizer_wrapper import ActionLatentTokenizerWrapper  # noqa: E402
from analyze_latents import _encode_mu_and_sample  # noqa: E402
from vsep_collect import build_datasets, sample_indices  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402
from matplotlib import cm  # noqa: E402


@torch.no_grad()
def gather(args, device):
    datasets, task_names = build_datasets(args)
    samp_task, samp_local = sample_indices(datasets, args)
    subsets = [torch.utils.data.Subset(datasets[ti], samp_local[samp_task == ti].tolist())
               for ti in range(len(datasets))]
    loader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset(subsets), batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=ActionFramesCollatorV4())
    wrap = ActionLatentTokenizerWrapper.from_checkpoint(args.v4_ckpt, device=device); wrap.eval()
    A, F0, F1, X0, X1 = [], [], [], [], []
    for b in loader:
        A.append(b["action"].float())
        f0, f1 = wrap._resolve_dino_feats(b["frame_x0"], b["frame_x1"], None, None, device)
        F0.append(f0.float().cpu()); F1.append(f1.float().cpu())
        X0.append(b["frame_x0"].cpu()); X1.append(b["frame_x1"].cpu())
    return (wrap, torch.cat(A), torch.cat(F0), torch.cat(F1),
            torch.cat(X0), torch.cat(X1), samp_task, task_names)


@torch.no_grad()
def enc_v4(wrap, a_row, F0, F1, device, batch=128):
    """encode a FIXED action a_row [T,D] with each donor's frames F0/F1[i]."""
    enc = wrap.tokenizer.encoder
    n = F0.shape[0]
    out = []
    for i in range(0, n, batch):
        bs = min(batch, n - i)
        a = a_row.unsqueeze(0).expand(bs, -1, -1).to(device)
        mu, *_ = _encode_mu_and_sample(enc, a, F0[i:i + bs].to(device), F1[i:i + bs].to(device))
        out.append(mu.float().cpu())
    return torch.cat(out)


@torch.no_grad()
def enc_v4_matched(wrap, A, F0, F1, device, batch=128):
    enc = wrap.tokenizer.encoder
    out = []
    for i in range(0, A.shape[0], batch):
        a = A[i:i + batch].to(device)
        mu, *_ = _encode_mu_and_sample(enc, a, F0[i:i + batch].to(device), F1[i:i + batch].to(device))
        out.append(mu.float().cpu())
    return torch.cat(out)


def img(x):
    x = np.asarray(x)
    if x.ndim == 3 and x.shape[0] in (1, 3) and x.shape[0] < x.shape[-1]:
        x = np.transpose(x, (1, 2, 0))
    return x.astype(np.uint8)


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
    ap.add_argument("--target-total", type=int, default=1500)
    ap.add_argument("--min-per-dataset", type=int, default=30)
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--radius-pct", type=float, default=0.7)
    ap.add_argument("--min-size", type=int, default=12)
    ap.add_argument("--n-groups", type=int, default=2)
    ap.add_argument("--n-pairs", type=int, default=7)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(_REPO_ROOT / "analysis" / "output" / "visual_sep_gr1"))
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    print("[swapviz] gathering ...")
    wrap4, A, F0, F1, X0, X1, samp_task, task_names = gather(args, device)
    N = A.shape[0]
    Af = A.reshape(N, -1).numpy()
    Xa = StandardScaler().fit_transform(Af)
    Vpool = torch.cat([F0.mean(1), F1.mean(1)], 1).numpy()   # [N, 2C] visual context

    # near-dup action groups (no task constraint), ranked by within-group visual spread
    rng = np.random.default_rng(args.sample_seed)
    i = rng.integers(0, N, 40000); j = rng.integers(0, N, 40000); ok = i != j
    radius = float(np.percentile(np.linalg.norm(Xa[i[ok]] - Xa[j[ok]], axis=1), args.radius_pct))
    nn = NearestNeighbors(radius=radius).fit(Xa)
    neigh = [np.asarray(m) for m in nn.radius_neighbors(Xa, return_distance=False) if len(m) >= args.min_size]
    neigh.sort(key=len, reverse=True)
    groups, used = [], np.zeros(N, bool)
    for mem in neigh:
        if used[mem].mean() > 0.5:
            continue
        groups.append(mem); used[mem] = True
        if len(groups) >= 12:
            break
    Vz = StandardScaler().fit_transform(Vpool)
    def vspread(m):
        P = Vz[m]; k = min(len(m), 100)
        idx = rng.choice(len(m), k, replace=False)
        d = [np.linalg.norm(P[a] - P[b]) for a in idx for b in idx if a < b]
        return np.mean(d)
    groups = sorted(groups, key=vspread, reverse=True)[:args.n_groups]
    print(f"[swapviz] radius p{args.radius_pct}={radius:.3f}; {len(groups)} groups")

    # v4 matched over ALL (for global-median normalization) + v3 wrapper
    Z4m = enc_v4_matched(wrap4, A, F0, F1, device, args.batch_size)
    sc4 = StandardScaler().fit(Z4m.reshape(N, -1).numpy())
    Z4m_s = sc4.transform(Z4m.reshape(N, -1).numpy())
    gi2 = rng.integers(0, N, 20000); gj = rng.integers(0, N, 20000); ok2 = gi2 != gj
    gmed4 = float(np.median(np.linalg.norm(Z4m_s[gi2[ok2]] - Z4m_s[gj[ok2]], axis=1)))

    wrap3 = ActionLatentTokenizerWrapper.from_checkpoint(args.v3_ckpt, device=device); wrap3.eval()
    dt3 = wrap3.tokenizer.encoder.action_proj.weight.dtype

    cmap = plt.get_cmap("turbo")
    for gi, mem in enumerate(groups):
        m = len(mem)
        # medoid action = member closest to the group's mean action
        Am = Af[mem]; center = Am.mean(0, keepdims=True)
        medoid = mem[int(np.argmin(np.linalg.norm(Am - center, axis=1)))]
        a_star = A[medoid]                                   # [T,D] FIXED action

        # FIXED action, swap each member's frames
        Z4 = enc_v4(wrap4, a_star, F0[mem], F1[mem], device, args.batch_size)   # [m,T,K]
        Z4s = sc4.transform(Z4.reshape(m, -1).numpy())
        # v3: fixed action → single latent (frame-independent)
        g, t, h = wrap3.encode(a_star.unsqueeze(0).to(device).to(dt3))
        z3_flat = t.reshape(1, -1).float().cpu().numpy()

        c4 = PCA(n_components=2, random_state=0).fit_transform(Z4s) / gmed4     # [m,2]
        v4_spread = float(np.mean([np.linalg.norm(c4[a] - c4[b]) for a in range(min(m,80)) for b in range(a+1, min(m,80))]))
        # color by visual axis
        vpc = PCA(n_components=2, random_state=0).fit_transform(Vz[mem])[:, 0]
        order = np.argsort(vpc); rank = np.empty(m); rank[order] = np.linspace(0, 1, m)
        colors = cmap(rank)
        K = min(args.n_pairs, m); sel = order[np.linspace(0, m - 1, K).astype(int)]

        L = 1.12 * max(np.abs(c4).max(), 1e-3)
        fig = plt.figure(figsize=(max(K * 1.55, 9.5), 9.2))
        gs = fig.add_gridspec(3, K, height_ratios=[3.3, 1.05, 1.05], hspace=0.28, wspace=0.08)
        ax3 = fig.add_subplot(gs[0, :max(1, K // 2)]); ax4 = fig.add_subplot(gs[0, K // 2:])
        # v3: single collapsed point
        ax3.scatter([0], [0], s=420, c="k", marker="o", zorder=3)
        ax3.annotate(f"all {m} chunks\n→ 1 identical latent", (0, 0), textcoords="offset points",
                     xytext=(0, -50), ha="center", fontsize=10)
        ax3.set_xlim(-L, L); ax3.set_ylim(-L, L); ax3.set_aspect("equal")
        ax3.set_title("v3 (action-only)   spread = 0.00", fontsize=12)
        ax3.grid(alpha=0.2); ax3.set_xticklabels([]); ax3.set_yticklabels([])
        ax3.set_ylabel("latent PC2 (global-median units)")
        # v4: spread cloud
        ax4.scatter(c4[:, 0], c4[:, 1], c=colors, s=32, alpha=0.85, linewidths=0)
        for r_i, li in enumerate(sel):
            ax4.scatter(c4[li, 0], c4[li, 1], s=190, facecolors="none", edgecolors="k", linewidths=1.7)
            ax4.annotate(str(r_i + 1), (c4[li, 0], c4[li, 1]), fontsize=9, fontweight="bold", ha="center", va="center")
        ax4.set_xlim(-L, L); ax4.set_ylim(-L, L); ax4.set_aspect("equal")
        ax4.set_title(f"v4 (DINO-fused)   spread = {v4_spread:.2f}", fontsize=12)
        ax4.grid(alpha=0.2); ax4.set_xticklabels([]); ax4.set_yticklabels([])

        for c, li in enumerate(sel):
            n = mem[li]
            for row, X in ((1, X0), (2, X1)):
                ax = fig.add_subplot(gs[row, c]); ax.imshow(img(X[n])); ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_edgecolor(colors[li]); sp.set_linewidth(3)
                if c == 0:
                    ax.set_ylabel("x0 (first)" if row == 1 else "x1 (last)", fontsize=9)
                if row == 1:
                    ax.set_title(f"pair {c+1}", fontsize=9, fontweight="bold", color=colors[li])
        fig.suptitle(
            f"②‴  ONE fixed action, {m} real visual contexts swapped in  (grp{gi})\n"
            f"v3 (action-only) puts ALL {m} at ONE point (spread=0)  ·  v4 (DINO-fused) spreads them "
            f"by visual (spread={v4_spread:.2f})\n"
            f"color = visual axis (DINO PC1);  numbered points ↔ x0/x1 frame pairs below "
            f"(motion same, scene differs)",
            fontsize=11.5, y=0.99)
        fig.tight_layout(rect=[0, 0, 1, 0.90])
        png = outdir / f"swapviz_grp{gi}.png"
        fig.savefig(png, dpi=125); plt.close(fig)
        print(f"[swapviz] grp{gi}: m={m} v4_spread={v4_spread:.3f} -> {png}")

    print("#### SWAPVIZ DONE ####")


if __name__ == "__main__":
    main()
