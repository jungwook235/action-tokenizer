#!/usr/bin/env python3
"""EXP-0010 pre-flight: measure the ACTION LATENT z scale on the trained prq tokenizer.

Why: vicreg's variance hinge targets "per-dim std >= 1", a threshold calibrated for the
reference implementation's DiT hidden. If our z's std is far below 1 the hinge is
saturated from step 0, the variance term dominates, and the regularizer's job silently
changes from "prevent collapse" to "inflate the latent". This measures the actual numbers
so --embod-reg-weight / --embod-reg-pool / the hinge threshold can be chosen, not guessed.

Reports, for both pooling modes (mean over the time axis vs every time token as a sample):
  * per-dim std distribution, per stream (human / robot) -- what the hinge sees
  * the human-vs-robot centroid gap ||mean_H - mean_R||^2 / d -- what invariance sees
  * the centroid estimator's variance floor at realistic batch sizes (gather or not)
  * the three vicreg terms at std targets 1.0 (reference) and s0 = median measured std

Usage:
  python zstats.py --ckpt <checkpoint-200000> --emb-config <embodiments.json> [-n 4096]
"""
import argparse
import json

import numpy as np
import torch

import gr00t.experiment.data_config_v3  # noqa: F401  (register extra configs)
from gr00t.data.dataset_action_frames_v4_multiemb import (
    EmbodimentTaggedDataset,
    MultiEmbActionFramesCollator,
)
from gr00t.data.dataset_egopi_prq_v4 import EgoPiPrqCachedDatasetV4
from gr00t.model.action_latent_tokenizer_v4_multiemb import MultiEmbActionLatentTokenizerV4

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--emb-config", required=True)
ap.add_argument("-n", "--n-samples", type=int, default=4096)
ap.add_argument("--batch-size", type=int, default=64)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--rec-pool", choices=["mean", "tokens"], default="tokens")
ap.add_argument("--rec-s0", type=float, default=1.75)
ap.add_argument("--rec-cov", type=float, default=0.004)
args = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
cfg = json.load(open(args.emb_config))
g = cfg["embodiments"][0]
name = g["name"]

# ---- datasets: exactly the training composition (size-proportional, no weights set) ----
dss = []
for src in g["sources"]:
    dss.append(EgoPiPrqCachedDatasetV4(
        prq_mode=src["mode"], prq_stats_path=g["prq_stats"], fk_cache_h5=src.get("fk_cache"),
        filter_json=g.get("filter"), filter_tag=src.get("filter_tag"),
        dataset_path=src["dataset_path"], data_config_name=src["data_config"],
        embodiment_tag=g.get("embodiment_tag", "new_embodiment"), split="train",
        val_ratio=0.003, val_seed=42, normalization_mode="min_max", image_size=224,
        feature_source="dino", dino_model="facebook/dinov2-large", dino_final_norm="naive",
        use_fixed_val=True, fixed_val_path=None, video_backend="decord"))
ds = EmbodimentTaggedDataset(torch.utils.data.ConcatDataset(dss), name)
print(f"[data] {len(g['sources'])} sources, {len(ds):,} train steps")

dl = torch.utils.data.DataLoader(
    ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
    collate_fn=MultiEmbActionFramesCollator(pass_is_human=True), drop_last=True,
    generator=torch.Generator().manual_seed(0))

# ---- model: the sbatch script's hyper-params + the checkpoint's class-token count ----
from safetensors.torch import load_file

sd = load_file(f"{args.ckpt}/model.safetensors", device="cpu")
n_pre = int(sd["embodiment_class_token"].shape[0])
action_dim = int(sd[f"action_encoders.{name}.action_proj.weight"].shape[1])
print(f"[ckpt] {args.ckpt}\n[ckpt] action_dim={action_dim} pretrain_class_tokens={n_pre}")
model = MultiEmbActionLatentTokenizerV4(
    embodiment_specs=[{"name": name, "action_dim": action_dim,
                       "class_token_id": int(g["class_token_id"])}],
    action_horizon=16, emb_dim=256, head_dim=64, encoder_depth=4, decoder_depth=4,
    decoder_mode="self_attention", token_dim=64, dino_dim=1024, fusion_width=1024,
    fusion_depth=6, fusion_heads=16, dino_decoder_depth=6,
    use_vae=True, vae_sample=False,  # deterministic mu -- what Stage-2 consumes
    lambda_recon=1.0, lambda_dino=0.1, lambda_kl=1e-6, recon_loss_type="l1",
    dino_loss_type="mse", dino_loss_weights={"l1": 0.0, "mse": 1.0, "cosine": 1.0},
    dino_final_norm="naive", use_embodiment_class_token=True,
    tokenizer_finetuning_mode=True, new_class_token=1, num_pretrain_class_tokens=n_pre,
)
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"[ckpt] loaded (missing={len(missing)} unexpected={len(unexpected)})")
assert not [m for m in missing if not m.startswith("_")], f"missing real params: {missing}"
model.eval().to(dev)

# ---- collect z ----
Z, L = [], []
seen = 0
with torch.no_grad():
    for b in dl:
        gg = b["groups"][name]
        z, _ = model.encode(name, gg["action"].to(dev),
                            gg["x0_feat"].to(dev), gg["x1_feat"].to(dev))
        Z.append(z.float().cpu())
        L.append(gg["is_human"].clone())
        seen += z.shape[0]
        if seen >= args.n_samples:
            break
Z = torch.cat(Z).double()          # [N, 16, 64]
L = torch.cat(L).double()
N, T, D = Z.shape
hm = L > 0.5
print(f"[z] N={N} T={T} d={D}  human={int(hm.sum())} robot={int((~hm).sum())}")
print(f"[z] raw element std (all N*T*d) = {Z.std().item():.4f}  "
      f"mean={Z.mean().item():+.4f}  abs-max={Z.abs().max().item():.3f}")


def q(v):
    v = np.sort(np.asarray(v))
    return (f"min {v[0]:.3f} | p10 {v[int(.1*len(v))]:.3f} | med {np.median(v):.3f} | "
            f"p90 {v[int(.9*len(v))]:.3f} | max {v[-1]:.3f}")


def vicreg_terms(H, R, s0, vic_var=1.0, vic_cov=0.04, inv_override=None):
    """The three terms exactly as EmbodAgnosticReg computes them, at std target s0."""
    d = H.shape[1]
    diff = H.mean(0) - R.mean(0)
    inv = float((diff * diff).sum() / d) if inv_override is None else float(inv_override)
    var = cov = 0.0
    for X in (H, R):
        std = torch.sqrt(X.var(0, unbiased=False) + 1e-4)
        var += float(torch.relu(s0 - std).pow(2).mean())
        Xc = X - X.mean(0, keepdim=True)
        C = (Xc.t() @ Xc) / (X.shape[0] - 1)
        off = C - torch.diag_embed(torch.diagonal(C))
        cov += float((off * off).sum() / d)
    return inv, vic_var * 0.5 * var, vic_cov * 0.5 * cov


def analyze(tag, h, lbl, bins=None):
    H, R = h[lbl > 0.5], h[lbl <= 0.5]
    sH = torch.sqrt(H.var(0, unbiased=False) + 1e-4)
    sR = torch.sqrt(R.var(0, unbiased=False) + 1e-4)
    print(f"\n===== {tag}  (N={h.shape[0]}, human={H.shape[0]}, robot={R.shape[0]}) =====")
    print(f"  per-dim std  human : {q(sH)}")
    print(f"  per-dim std  robot : {q(sR)}")
    frac = float(((sH < 1.0).float().mean() + (sR < 1.0).float().mean()) / 2)
    print(f"  dims with std < 1.0 (hinge ACTIVE at the reference threshold): {100*frac:.1f}%")
    diff = H.mean(0) - R.mean(0)
    gap = float((diff * diff).sum() / h.shape[1])
    print(f"  H/R centroid gap ||dmu||^2/d = {gap:.5f}   (||dmu|| = {float(diff.norm()):.4f})")

    # Estimator variance floor: E[||mu_H_hat - mu_R_hat||^2]/d = gap + (trS_H/n_h + trS_R/n_r)/d
    trH = float(H.var(0, unbiased=False).sum())
    trR = float(R.var(0, unbiased=False).sum())
    for label, nh, nr in (("per-rank micro-batch (no gather, 64 @ ~47/53)", 30, 34),
                          ("8-GPU all-gather (512 @ ~47/53)", 241, 271)):
        floor = (trH / nh + trR / nr) / h.shape[1]
        print(f"  estimator floor, {label}: {floor:.5f} "
              f"({floor/max(gap,1e-12):.1f}x the true gap)")

    s_med = float(torch.median(torch.cat([sH, sR])))
    for s0, nm in ((1.0, "s0=1.0 (reference)"), (s_med, f"s0={s_med:.3f} (median measured)")):
        inv, var, cov = vicreg_terms(H, R, s0, inv_override=None)
        tot = inv + var + cov
        print(f"  vicreg @ {nm:<28} inv={inv:.5f}  var={var:.5f}  cov={cov:.5f}  "
              f"total={tot:.5f}   var/inv={var/max(inv,1e-12):.1f}x")
    return gap, s_med


print("\n" + "#" * 78)
print("# 1) pool=mean  -- z averaged over the 16 time tokens")
print("#" * 78)
gap_m, smed_m = analyze("pool=mean", Z.mean(1), L)

print("\n" + "#" * 78)
print("# 2) pool=tokens -- every time token is its own sample (bin = token index)")
print("#" * 78)
Zt = Z.reshape(N * T, D)
Lt = L.repeat_interleave(T)
gap_t, smed_t = analyze("pool=tokens (global)", Zt, Lt)

# Stratified invariance: the within-bin contrast our _stratified_inv actually computes.
bins = torch.arange(T).repeat(N)
num = den = 0.0
per_bin = []
for b in range(T):
    hb = (Lt > 0.5) & (bins == b)
    rb = (Lt <= 0.5) & (bins == b)
    dmu = Zt[hb].mean(0) - Zt[rb].mean(0)
    term = float((dmu * dmu).sum() / D)
    w = float(min(int(hb.sum()), int(rb.sum())))
    num += w * term
    den += w
    per_bin.append(term)
print(f"\n  stratified invariance (per-token-bin, min-count weighted) = {num/den:.5f}")
print(f"  per-bin gap: t0 {per_bin[0]:.4f}  t7 {per_bin[7]:.4f}  t15 {per_bin[15]:.4f}  "
      f"(min {min(per_bin):.4f} / max {max(per_bin):.4f})")

# ---- exact value of the loss under a candidate config, so the weight can be derived ----
print("\n" + "#" * 78)
print(f"# 3) candidate config: pool={args.rec_pool} vic_std={args.rec_s0} "
      f"vic_cov={args.rec_cov}")
print("#" * 78)
if args.rec_pool == "tokens":
    hh, ll, inv_use = Zt, Lt, num / den   # stratified invariance
else:
    hh, ll, inv_use = Z.mean(1), L, None
inv, var, cov = vicreg_terms(hh[ll > 0.5], hh[ll <= 0.5], args.rec_s0,
                             vic_cov=args.rec_cov, inv_override=inv_use)
tot = inv + var + cov
print(f"  inv={inv:.5f} ({100*inv/tot:.0f}%)  var={var:.5f} ({100*var/tot:.0f}%)  "
      f"cov={cov:.5f} ({100*cov/tot:.0f}%)   loss_embod_reg = {tot:.5f}")
for w in (0.001, 0.002, 0.003, 0.005, 0.01, 0.1):
    print(f"    weight {w:<6} -> contribution {w*tot:.6f}")

# Reference scale for choosing the weight: the run's own recon/dino losses.
try:
    ts = json.load(open(f"{args.ckpt}/trainer_state.json"))
    last = [h for h in ts.get("log_history", []) if "loss" in h][-1]
    print(f"\n[scale] final logged train losses @ step {last.get('step')}: "
          + "  ".join(f"{k}={v:.5f}" for k, v in last.items()
                      if k.startswith("loss") and isinstance(v, float)))
except Exception as e:  # noqa: BLE001
    print(f"\n[scale] trainer_state.json unreadable: {e}")
