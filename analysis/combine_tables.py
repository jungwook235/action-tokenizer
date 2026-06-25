"""Combine the five per-tokenizer latent summaries into ONE comparison table.

Reads the saved arrays (``*_latent_summary.npz``: mu/z/sigma/logvar) and parses a
few scalars (action_dim, recon L1, val chunk counts, data_config) from the matching
``*_latent_summary.txt``, then writes a single side-by-side comparison txt.

Run from the action_tokenizer repo root (no GPU needed):
    python analysis/combine_tables.py
"""

import re
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).resolve().parent / "output"

# (column header, npz basename, human label). Order = column order.
TOKENIZERS = [
    ("dexjoco_v4vae", "dexjoco_dual_arm_latent_summary",
     "dexjoco_dual_arm  (V4 + VAE)"),
    ("gr1_1k_v4vae", "gr1_1000demos_latent_summary",
     "gr1_1000demos  (V4 + VAE)"),
    ("gr1_100_v4vae", "gr1_100demos_v4_vae_latent_summary",
     "gr1_100demos  (V4 + VAE)"),
    ("gr1_100_v4", "gr1_100demos_v4_novae_latent_summary",
     "gr1_100demos  (V4, no VAE)"),
    ("gr1_1k_v3", "gr1_1000demos_v3_latent_summary",
     "gr1_1000demos  (V3, K=16)"),
]


def parse_txt(txt: str) -> dict:
    def grab(pat, cast=str, default=None):
        m = re.search(pat, txt)
        return cast(m.group(1)) if m else default
    d = {}
    d["data_config"] = grab(r"data_config\s*:\s*(\S+)")
    d["ver"] = grab(r"tokenizer_type\s*:\s*(.+)")
    d["action_dim"] = grab(r"action_dim \(D\)\s*:\s*(\d+)", int)
    d["T"] = grab(r"action_horizon \(T\)\s*:\s*(\d+)", int)
    d["n_datasets"] = grab(r"datasets \((\d+)\)", int)
    d["total_chunks"] = grab(r"total val chunks available\s*:\s*(\d+)", int)
    d["analyzed"] = grab(r"chunks analyzed\s*:\s*(\d+)", int)
    # recon L1: VAE files have decode(μ)/decode(z); deterministic has decode(latent)
    d["recon_mu"] = grab(r"decode\(μ\) vs GT = ([\d.]+)", float)
    d["recon_z"] = grab(r"decode\(z\) vs GT = ([\d.]+)", float)
    d["recon_det"] = grab(r"decode\(latent\) vs GT = ([\d.]+)", float)
    return d


def stats_from_npz(npz) -> dict:
    mu = npz["mu"].astype(np.float64)          # [N, T, K]
    z = npz["z"].astype(np.float64)
    sigma = npz["sigma"].astype(np.float64)
    N, T, K = mu.shape
    is_vae = float(sigma.max()) > 0.0
    tok_norm = np.linalg.norm(z.reshape(-1, K), axis=-1)
    chunk_norm = np.linalg.norm(z.reshape(N, -1), axis=-1)
    noise = z - mu
    out = {
        "N": N, "T": T, "K": K, "latent_per_chunk": T * K, "is_vae": is_vae,
        "mu_mean": mu.mean(), "mu_std": mu.std(),
        "mu_rms": np.sqrt((mu ** 2).mean()),
        "mu_min": mu.min(), "mu_max": mu.max(), "mu_absmean": np.abs(mu).mean(),
        "z_rms": np.sqrt((z ** 2).mean()),
        "tok_l2_mean": tok_norm.mean(), "chunk_l2_mean": chunk_norm.mean(),
    }
    if is_vae:
        out["sigma_mean"] = sigma.mean()
        out["noise_rms"] = np.sqrt((noise ** 2).mean())
        out["noise_to_signal"] = out["noise_rms"] / max(1e-9, out["mu_rms"])
        kl = (-0.5 * (1.0 + npz["logvar"] - npz["mu"] ** 2 - np.exp(npz["logvar"])))
        out["kl_total"] = float(kl.reshape(-1, K).mean(0).sum())
        out["active_dims"] = int((kl.reshape(-1, K).mean(0) > 0.01).sum())
    return out


def fmt(v, p=4):
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):d}"
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.{p}f}"
    return str(v)


def main():
    cols = []
    for key, base, label in TOKENIZERS:
        npz_path = OUT_DIR / f"{base}.npz"
        txt_path = OUT_DIR / f"{base}.txt"
        if not npz_path.exists():
            print(f"[skip] missing {npz_path}")
            continue
        s = stats_from_npz(np.load(npz_path))
        t = parse_txt(txt_path.read_text()) if txt_path.exists() else {}
        recon = t.get("recon_z") if s["is_vae"] else t.get("recon_det")
        cols.append((key, label, s, t, recon))

    # Build rows: list of (row_label, value_fn(s,t,recon))
    rows = [
        ("VAE sampling", lambda s, t, r: "yes" if s["is_vae"] else "no(z==μ)"),
        ("action_dim D", lambda s, t, r: t.get("action_dim")),
        ("horizon T", lambda s, t, r: s["T"]),
        ("token_dim K", lambda s, t, r: s["K"]),
        ("latent / chunk (T·K)", lambda s, t, r: s["latent_per_chunk"]),
        ("val chunks (total)", lambda s, t, r: t.get("total_chunks")),
        ("chunks analyzed", lambda s, t, r: s["N"]),
        ("__sep__", None),
        ("μ mean", lambda s, t, r: s["mu_mean"]),
        ("μ std", lambda s, t, r: s["mu_std"]),
        ("μ RMS", lambda s, t, r: s["mu_rms"]),
        ("μ abs-mean", lambda s, t, r: s["mu_absmean"]),
        ("μ min", lambda s, t, r: s["mu_min"]),
        ("μ max", lambda s, t, r: s["mu_max"]),
        ("z RMS", lambda s, t, r: s["z_rms"]),
        ("token L2 ‖z‖ (mean)", lambda s, t, r: s["tok_l2_mean"]),
        ("chunk L2 ‖z‖ (mean)", lambda s, t, r: s["chunk_l2_mean"]),
        ("__sep__", None),
        ("σ mean (VAE)", lambda s, t, r: s.get("sigma_mean")),
        ("noise RMS (z-μ)", lambda s, t, r: s.get("noise_rms")),
        ("noise/signal", lambda s, t, r: s.get("noise_to_signal")),
        ("KL total /token", lambda s, t, r: s.get("kl_total")),
        ("active dims (/K)", lambda s, t, r: s.get("active_dims")),
        ("__sep__", None),
        ("recon L1 (z) vs GT", lambda s, t, r: r),
    ]

    label_w = 22
    col_w = 16
    lines = []
    W = label_w + 1 + (col_w + 1) * len(cols)
    lines.append("=" * W)
    lines.append("ACTION-LATENT TOKENIZER COMPARISON  (5 tokenizers, validation sets)")
    lines.append("=" * W)
    lines.append("Latent recorded as pre-sampling μ and post-sampling z (z==μ for the")
    lines.append("deterministic / non-VAE tokenizers). Stats over all analyzed val chunks.")
    lines.append("")
    # legend
    for key, label, *_ in cols:
        lines.append(f"  {key:<16} = {label}")
    lines.append("")

    # header
    head = f"{'metric':<{label_w}}│" + "│".join(f"{k:^{col_w}}" for k, *_ in cols)
    lines.append(head)
    lines.append("─" * label_w + "┼" + "┼".join("─" * col_w for _ in cols))
    for rlabel, fn in rows:
        if rlabel == "__sep__":
            lines.append("─" * label_w + "┼" + "┼".join("─" * col_w for _ in cols))
            continue
        cells = []
        for key, label, s, t, recon in cols:
            cells.append(f"{fmt(fn(s, t, recon)):^{col_w}}")
        lines.append(f"{rlabel:<{label_w}}│" + "│".join(cells))
    lines.append("=" * W)
    lines.append("")
    lines.append("Notes")
    lines.append("  • VAE σ≈0.018–0.019 everywhere → sampling perturbs the latent <1% (z≈μ).")
    lines.append("  • The tiny KL (λ_kl=1e-6) regularizes scale: V4+VAE μ RMS≈1.9–2.6, while")
    lines.append("    V4 without VAE has no such pressure (μ RMS≈5.3). V3 (K=16) μ RMS≈1.9.")
    lines.append("  • Reconstruction L1 is ~equally low (~0.0016–0.005) across all tokenizers.")

    out = OUT_DIR / "ALL_tokenizers_latent_comparison.txt"
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\n[combine] wrote -> {out}")


if __name__ == "__main__":
    main()
