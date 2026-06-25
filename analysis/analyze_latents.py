"""Analyze the final action latent produced by an action-latent tokenizer on its
*validation* set. Supports:

  * V4 (RLA-DINO hybrid), with or without the SD-style VAE bottleneck.
  * V3 / V2 (action-only, LayerNorm bottleneck), which are deterministic.

For a V4-VAE tokenizer the fusion encoder output is the posterior mean ``mu``;
``encode`` then reparameterizes ``z = mu + sigma * eps`` (sigma = exp(0.5*logvar)),
and the downstream VLA target is the *sampled* ``z``. This script records the
latent **just before sampling (mu)** and **just after sampling (z)** as a pair,
plus the sampling noise (sigma / logvar / KL). For deterministic tokenizers
(V4 without --use-vae, or V3/V2) there is NO sampling, so z == mu and the report
says so explicitly.

It reuses the project's own loaders so the numbers match training exactly:
  * ``ActionLatentTokenizerWrapper.from_checkpoint`` (architecture auto-detected
    from the state_dict; for V4 it builds the frozen dinov2-large extractor),
  * ``ActionFramesDatasetV4`` (split="val", same fixed-val split as training),
  * ``apply_merged_normalization_metadata`` (same whole-mixture action norm).

Self-contained: lives under ``analysis/`` and does NOT modify any other code.
For V4 the latent is recomputed locally (``_encode_mu_and_sample``) as a faithful
re-implementation of ``TimeWiseEncoderV4.forward`` so BOTH mu and z are captured
from one forward (the public ``encode`` returns only z). For V3/V2 the public
``encode(actions)`` already returns the (deterministic) latent.
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Make the repo root importable regardless of cwd (script lives in analysis/).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import gr00t.experiment.data_config_v3  # noqa: F401  (registers extra data configs)
from gr00t.data.dataset_action_frames_v4 import (  # noqa: E402
    ActionFramesCollatorV4,
    ActionFramesDatasetV4,
)
from gr00t.data.merge_norm_stats import apply_merged_normalization_metadata  # noqa: E402
from gr00t.model.action_latent_tokenizer_wrapper import (  # noqa: E402
    ActionLatentTokenizerWrapper,
)


# =====================================================================
# Faithful re-implementation of TimeWiseEncoderV4.forward that also
# returns mu / sigma / logvar (the public V4 forward returns only z).
# =====================================================================


@torch.no_grad()
def _encode_mu_and_sample(encoder, actions, x0_feat, x1_feat, generator=None):
    """V4 only. Return (mu, sigma, logvar, z), each [B, T, token_dim].

    Mirrors ``TimeWiseEncoderV4.forward`` exactly:
      action_encoder -> act_tokens -> joint(out_layer = bottleneck) = mu
      logvar = logvar_head(mu).clamp(...); sigma = exp(0.5*logvar); z = mu + eps*sigma
    With num_global_tokens = num_hand_tokens = 0, every token is a time token,
    so the returned tensors are the full [B, T, token_dim] action latent.
    """
    dtype = encoder.action_proj.weight.dtype
    actions = actions.to(dtype=dtype)
    dino_diff = x1_feat.to(dtype=dtype) - x0_feat.to(dtype=dtype)

    g256, t256, h256 = encoder.action_encoder(actions)
    act_tokens = torch.cat([g256, t256, h256], dim=1)
    tokens_out, _ = encoder.joint(x=dino_diff, tokens=act_tokens)  # [B, T, token_dim]

    mu = tokens_out
    if getattr(encoder, "use_vae", False):
        logvar = encoder.logvar_head(mu).clamp(encoder.kl_logvar_min, encoder.kl_logvar_max)
        sigma = torch.exp(0.5 * logvar)
        eps = torch.empty_like(sigma).normal_(generator=generator)
        z = mu + eps * sigma
    else:
        logvar = torch.zeros_like(mu)
        sigma = torch.zeros_like(mu)
        z = mu
    return mu.float(), sigma.float(), logvar.float(), z.float()


# =====================================================================
# Formatting helpers (box headers + aligned tables).
# =====================================================================


def _hline(width=80, ch="="):
    return ch * width


def _box_title(title, width=92):
    inner = width - 2
    return "\n".join([
        "╔" + "═" * inner + "╗",
        "║" + title.center(inner) + "║",
        "╚" + "═" * inner + "╝",
    ])


def _fmt(x, p=6):
    if x is None:
        return "NA"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):d}"
    return f"{float(x):.{p}f}"


def _stat_block(name, t):
    flat = t.reshape(-1).float()
    if flat.numel() <= 16_000_000:
        q = torch.quantile(flat, torch.tensor([0.01, 0.5, 0.99]))
    else:
        q = torch.quantile(flat[torch.randperm(flat.numel())[:16_000_000]],
                           torch.tensor([0.01, 0.5, 0.99]))
    return {
        "name": name, "mean": flat.mean().item(), "std": flat.std().item(),
        "rms": flat.pow(2).mean().sqrt().item(), "min": flat.min().item(),
        "max": flat.max().item(), "absmean": flat.abs().mean().item(),
        "p01": q[0].item(), "p50": q[1].item(), "p99": q[2].item(),
    }


def _two_col_table(left, right, label_w=22, col_w=18):
    rows = [
        ("elements mean", "mean"), ("elements std", "std"), ("RMS (sqrt<x^2>)", "rms"),
        ("abs mean", "absmean"), ("min", "min"), ("max", "max"),
        ("p01", "p01"), ("p50 (median)", "p50"), ("p99", "p99"),
    ]
    out = [f"{'statistic':<{label_w}}│{left['name']:^{col_w}}│{right['name']:^{col_w}}"]
    out.append("─" * label_w + "┼" + "─" * col_w + "┼" + "─" * col_w)
    for disp, key in rows:
        out.append(f"{disp:<{label_w}}│{_fmt(left[key]):^{col_w}}│{_fmt(right[key]):^{col_w}}")
    return "\n".join(out)


# =====================================================================
# Data
# =====================================================================


def build_val_dataset(args):
    def make(path, split):
        return ActionFramesDatasetV4(
            dataset_path=path, data_config_name=args.data_config,
            embodiment_tag=args.embodiment_tag, split=split,
            val_ratio=args.val_ratio, val_seed=args.val_seed,
            normalization_mode=args.normalization_mode, image_size=args.image_size,
            video_backend=args.video_backend, use_fixed_val=args.use_fixed_val,
            fixed_val_path=args.fixed_val_path,
        )

    datasets_val, per_dataset_info = [], []
    for path in args.dataset_path:
        assert os.path.exists(path), f"Dataset path does not exist: {path}"
        dva = make(path, "val")
        datasets_val.append(dva)
        per_dataset_info.append((Path(path).name, len(dva.all_steps)))

    # Replicate training-time normalization. Both gr1/dexjoco configs normalize
    # every action key with "min_max", whose merged min/max/q01/q99 are weight-
    # INDEPENDENT and read from each dataset's full-dataset metadata (split-
    # independent). So merging from the val datasets alone yields the IDENTICAL
    # applied normalization, while avoiding building the very large train indices.
    apply_merged_normalization_metadata(datasets_val, datasets_val)

    val_dataset = datasets_val[0] if len(datasets_val) == 1 \
        else torch.utils.data.ConcatDataset(datasets_val)
    return val_dataset, per_dataset_info


# =====================================================================
# Main
# =====================================================================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-config", required=True)
    ap.add_argument("--dataset-path", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--embodiment-tag", default="new_embodiment")
    ap.add_argument("--normalization-mode", default="min_max")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--video-backend", default="decord")
    ap.add_argument("--val-ratio", type=float, default=0.003)
    ap.add_argument("--val-seed", type=int, default=42)
    ap.add_argument("--use-fixed-val", action="store_true", default=True)
    ap.add_argument("--fixed-val-path", default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--max-samples", type=int, default=4096,
                    help="cap on number of action chunks analyzed (-1 = all)")
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--num-examples", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.sample_seed)
    print(f"[analyze] tag={args.tag} device={device}")
    print(f"[analyze] checkpoint={args.checkpoint}")

    # ---- model ----
    wrapper = ActionLatentTokenizerWrapper.from_checkpoint(args.checkpoint, device=device)
    wrapper.eval()
    tok = wrapper.tokenizer
    encoder = tok.encoder

    is_v5 = hasattr(tok, "_is_v5")
    is_v4 = hasattr(tok, "_is_v4")
    is_v3 = hasattr(tok, "_is_v3")
    is_v2 = hasattr(tok, "_is_v2") and not is_v3
    needs_visual = is_v4 or is_v5
    use_vae = bool(getattr(encoder, "use_vae", False))
    if is_v5:
        raise NotImplementedError("V5 (LAM) analysis is not covered by this script.")
    if is_v4:
        ver_str = "V4 (RLA-DINO hybrid)" + (" + SD-style VAE" if use_vae else " (deterministic, no VAE)")
    elif is_v3:
        ver_str = "V3 (action-only, LayerNorm bottleneck; deterministic)"
    elif is_v2:
        ver_str = "V2 (action-only; deterministic)"
    else:
        ver_str = "timewise (deterministic)"
    print(f"[analyze] tokenizer={ver_str} use_vae={use_vae} needs_visual={needs_visual}")

    # ---- data ----
    val_dataset, per_dataset_info = build_val_dataset(args)
    n_total_chunks = sum(n for _, n in per_dataset_info)
    loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=ActionFramesCollatorV4(), drop_last=False,
    )

    # ---- run encode over the val set ----
    gen = torch.Generator(device=device).manual_seed(args.sample_seed)
    mus, zs, sigmas, logvars = [], [], [], []
    recon_l1_mu = recon_l1_z = 0.0
    n_seen = 0
    cap = args.max_samples if args.max_samples > 0 else float("inf")

    for batch in loader:
        if n_seen >= cap:
            break
        actions = batch["action"].to(device)

        if needs_visual:
            f0, f1 = wrapper._resolve_dino_feats(
                batch["frame_x0"], batch["frame_x1"], None, None, device)
            mu, sigma, logvar, z = _encode_mu_and_sample(encoder, actions, f0, f1, generator=gen)
            zero_g = mu[:, :0]
            rec_mu = tok.decode(zero_g, mu.to(encoder.action_proj.weight.dtype), zero_g)
            rec_z = tok.decode(zero_g, z.to(encoder.action_proj.weight.dtype), zero_g)
        else:
            # V3 / V2: deterministic, action-only encode → (global, time, hand).
            g, t, h = tok.encode(actions.to(encoder.action_proj.weight.dtype))
            mu = t.float()
            sigma = torch.zeros_like(mu)
            logvar = torch.zeros_like(mu)
            z = mu.clone()
            rec = tok.decode(g, t, h)
            rec_mu = rec_z = rec

        a = actions.to(rec_mu.dtype)
        bs = actions.shape[0]
        recon_l1_mu += torch.nn.functional.l1_loss(rec_mu, a).item() * bs
        recon_l1_z += torch.nn.functional.l1_loss(rec_z, a).item() * bs

        mus.append(mu.cpu()); zs.append(z.cpu())
        sigmas.append(sigma.cpu()); logvars.append(logvar.cpu())
        n_seen += bs

    mu = torch.cat(mus)
    if cap != float("inf"):
        mu = mu[: int(cap)]
    z = torch.cat(zs)[: mu.shape[0]]
    sigma = torch.cat(sigmas)[: mu.shape[0]]
    logvar = torch.cat(logvars)[: mu.shape[0]]
    n_used = mu.shape[0]
    recon_l1_mu /= max(1, n_seen)
    recon_l1_z /= max(1, n_seen)

    B, T, K = mu.shape
    print(f"[analyze] latent tensor (chunks,T,token_dim)={tuple(mu.shape)}")

    # ---- norms ----
    tok_norm_mu = mu.reshape(-1, K).norm(dim=-1)
    tok_norm_z = z.reshape(-1, K).norm(dim=-1)
    chunk_norm_mu = mu.reshape(B, -1).norm(dim=-1)
    chunk_norm_z = z.reshape(B, -1).norm(dim=-1)

    # ---- per-dim structure ----
    flat_mu = mu.reshape(-1, K)
    flat_sigma = sigma.reshape(-1, K)
    flat_logvar = logvar.reshape(-1, K)
    dim_mu_mean = flat_mu.mean(0)
    dim_mu_std = flat_mu.std(0)
    dim_sigma_mean = flat_sigma.mean(0)
    if use_vae:
        kl_dim = (-0.5 * (1.0 + flat_logvar - flat_mu.pow(2) - flat_logvar.exp())).mean(0)
        total_kl = kl_dim.sum().item()
        kl_thresh = 0.01
        active = (kl_dim > kl_thresh).sum().item()
    else:
        kl_dim = None
    noise = (z - mu)
    noise_rms = noise.pow(2).mean().sqrt().item()

    # ---- examples ----
    n_ex = min(args.num_examples, B)
    ex_mu = mu[:n_ex, 0, :]
    ex_z = z[:n_ex, 0, :]
    ex_sigma = sigma[:n_ex, 0, :]

    # ---- report ----
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    W = 100
    L = [_hline(W), f"Action-latent analysis  —  tokenizer: {args.tag}", _hline(W)]
    L.append(f"generated_at   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"checkpoint     : {args.checkpoint}")
    L.append(f"data_config    : {args.data_config}")
    L.append(f"tokenizer_type : {ver_str}")
    if needs_visual:
        L.append(f"dino_extractor : facebook/dinov2-large (frozen), dino_final_norm="
                 f"{getattr(tok, 'dino_final_norm', 'affine')}")
    L.append("")
    L.append("Latent definition")
    if use_vae:
        L.append("  pre-sampling  μ  : fusion-encoder bottleneck output (posterior mean)")
        L.append("  post-sampling z  : μ + σ·ε,  σ = exp(0.5·logvar_head(μ)),  ε~N(0,I)")
        L.append("  → the VLA training target is the sampled z.")
    else:
        L.append("  latent           : encoder bottleneck output (deterministic)")
        L.append("  NO VAE sampling  : z == μ  (the μ vs z columns below are identical by construction)")
    L.append("")

    L.append(_box_title("LATENT SHAPE & ARCHITECTURE", W))
    L.append(f"  action_dim (D)        : {wrapper.action_dim}")
    L.append(f"  action_horizon (T)    : {wrapper.action_horizon}")
    L.append(f"  token_dim (latent K)  : {K}")
    L.append(f"  latent per chunk      : [T={T}, K={K}]  → {T*K} values "
             f"(num_main_tokens={wrapper.num_main_tokens}, Ng={wrapper.num_global_tokens}, "
             f"Nh={wrapper.num_hand_tokens})")
    L.append(f"  internal_emb_dim      : {wrapper.internal_emb_dim}  (transformer width)")
    L.append("")

    L.append(_box_title("VALIDATION SET", W))
    L.append(f"  fixed-val split (val_seed={args.val_seed}, val_ratio={args.val_ratio})")
    if args.fixed_val_path:
        L.append(f"  fixed_val_path : {args.fixed_val_path}")
    L.append(f"  datasets ({len(per_dataset_info)}):")
    for name, n in per_dataset_info:
        L.append(f"      {name:<70} chunks={n}")
    L.append(f"  total val chunks available : {n_total_chunks}")
    capped = (args.max_samples > 0 and n_total_chunks > args.max_samples)
    L.append(f"  chunks analyzed            : {n_used}"
             + (f"  (capped at --max-samples={args.max_samples})" if capped else ""))
    L.append("")

    L.append(_box_title("LATENT MAGNITUDE / SCALE  —  μ (pre-sample)  vs  z (post-sample)", W))
    L.append(_two_col_table(_stat_block("μ (pre)", mu), _stat_block("z (post)", z)))
    s_mu_rms = mu.reshape(-1).pow(2).mean().sqrt().item()
    L.append("")
    L.append("  L2 norm of one latent token vector (dim K):")
    L.append(f"      μ : mean={_fmt(tok_norm_mu.mean().item())}  "
             f"min={_fmt(tok_norm_mu.min().item())}  max={_fmt(tok_norm_mu.max().item())}")
    L.append(f"      z : mean={_fmt(tok_norm_z.mean().item())}  "
             f"min={_fmt(tok_norm_z.min().item())}  max={_fmt(tok_norm_z.max().item())}")
    L.append("  L2 norm of the whole-chunk latent (T·K values):")
    L.append(f"      μ : mean={_fmt(chunk_norm_mu.mean().item())}  "
             f"min={_fmt(chunk_norm_mu.min().item())}  max={_fmt(chunk_norm_mu.max().item())}")
    L.append(f"      z : mean={_fmt(chunk_norm_z.mean().item())}  "
             f"min={_fmt(chunk_norm_z.min().item())}  max={_fmt(chunk_norm_z.max().item())}")
    L.append("")

    if use_vae:
        L.append(_box_title("VAE SAMPLING  (the gap between μ and z)", W))
        L.append("  σ = exp(0.5·logvar)  [per element]:")
        L.append(f"      mean={_fmt(sigma.mean().item())}  min={_fmt(sigma.min().item())}  "
                 f"max={_fmt(sigma.max().item())}  median={_fmt(sigma.reshape(-1).median().item())}")
        L.append(f"  logvar : mean={_fmt(logvar.mean().item())}  min={_fmt(logvar.min().item())}  "
                 f"max={_fmt(logvar.max().item())}")
        L.append(f"  sampling noise (z-μ) RMS : {_fmt(noise_rms)}   "
                 f"(≈ σ; compare to μ RMS={_fmt(s_mu_rms)})")
        L.append(f"  noise-to-signal  RMS(z-μ)/RMS(μ) : {_fmt(noise_rms / max(1e-9, s_mu_rms))}")
        L.append(f"  KL(q‖N(0,I))  total per token  : {_fmt(total_kl, 4)} nats over {K} dims")
        L.append(f"  active latent dims (KL>{kl_thresh}) : {active} / {K}")
        L.append("")

    # per-dim table
    L.append(_box_title(f"PER-DIM LATENT STRUCTURE  ({K} dims, over all tokens)", W))
    if use_vae:
        L.append(f"  {'dim':>3} │ {'μ_mean':>10} {'μ_std':>10} │ {'σ_mean':>10} │ {'KL':>9}")
        L.append("  " + "─" * 4 + "┼" + "─" * 23 + "┼" + "─" * 12 + "┼" + "─" * 10)
        order = torch.argsort(kl_dim, descending=True)
        for d in order.tolist():
            L.append(f"  {d:>3} │ {_fmt(dim_mu_mean[d].item()):>10} {_fmt(dim_mu_std[d].item()):>10} │ "
                     f"{_fmt(dim_sigma_mean[d].item()):>10} │ {_fmt(kl_dim[d].item(),4):>9}")
        L.append("  (rows sorted by KL = most-used latent dims first)")
    else:
        L.append(f"  {'dim':>3} │ {'μ_mean':>12} {'μ_std':>12} {'|μ|_max':>12}")
        L.append("  " + "─" * 4 + "┼" + "─" * 40)
        dim_mu_absmax = flat_mu.abs().max(0).values
        order = torch.argsort(dim_mu_std, descending=True)
        for d in order.tolist():
            L.append(f"  {d:>3} │ {_fmt(dim_mu_mean[d].item()):>12} {_fmt(dim_mu_std[d].item()):>12} "
                     f"{_fmt(dim_mu_absmax[d].item()):>12}")
        L.append("  (rows sorted by μ_std = highest-variance latent dims first)")
    L.append("")

    L.append(_box_title(f"EXAMPLE LATENTS  (first {n_ex} val chunks, time token t=0)", W))
    if use_vae:
        L.append("  Each example shows the pre-sampling μ and post-sampling z for the SAME chunk.")
    else:
        L.append("  Deterministic tokenizer: z == μ, so only μ is shown.")
    show_k = min(K, 16)
    for i in range(n_ex):
        if use_vae:
            L.append(f"  ── example #{i}  (‖μ‖={_fmt(ex_mu[i].norm().item(),3)}, "
                     f"‖z‖={_fmt(ex_z[i].norm().item(),3)}) ──")
        else:
            L.append(f"  ── example #{i}  (‖μ‖={_fmt(ex_mu[i].norm().item(),3)}) ──")
        L.append(f"      μ[:{show_k}] = " + " ".join(f"{v:+.3f}" for v in ex_mu[i, :show_k].tolist()))
        if use_vae:
            L.append(f"      z[:{show_k}] = " + " ".join(f"{v:+.3f}" for v in ex_z[i, :show_k].tolist()))
            L.append(f"      σ[:{show_k}] = " + " ".join(f"{v:.3f}" for v in ex_sigma[i, :show_k].tolist()))
    L.append("")

    L.append(_box_title("DECODE SANITY (bonus)", W))
    if use_vae:
        L.append(f"  action recon L1 (normalized space):  decode(μ) vs GT = {_fmt(recon_l1_mu)}")
        L.append(f"  action recon L1 (normalized space):  decode(z) vs GT = {_fmt(recon_l1_z)}")
        L.append("  (z is the VLA target; gap shows how much VAE sampling perturbs reconstruction.)")
    else:
        L.append(f"  action recon L1 (normalized space):  decode(latent) vs GT = {_fmt(recon_l1_mu)}")
    L.append("")
    L.append(_hline(W))

    report = "\n".join(L)
    out_path.write_text(report)
    npz = out_path.with_suffix(".npz")
    np.savez_compressed(
        npz, mu=mu.numpy(), z=z.numpy(), sigma=sigma.numpy(), logvar=logvar.numpy(),
        dim_mu_mean=dim_mu_mean.numpy(), dim_mu_std=dim_mu_std.numpy(),
        dim_sigma_mean=dim_sigma_mean.numpy(),
        kl_dim=(kl_dim.numpy() if use_vae else np.zeros(K, dtype=np.float32)),
    )
    print(report)
    print(f"\n[analyze] wrote report -> {out_path}")
    print(f"[analyze] wrote arrays -> {npz}")


if __name__ == "__main__":
    main()
