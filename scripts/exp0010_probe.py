#!/usr/bin/env python3
"""EXP-0010 CPU smoke: 4 configurations of the multiemb V4 tokenizer.

  off  : baseline (both features off)  -> the byte-identity reference
  A    : embod_reg only (vicreg, pool=mean) and a pool=tokens variant
  B    : per-domain recon decoder split only
  AB   : both

Every configuration loads the SAME shared weights (the "off" model's state_dict) before
the forward, exactly as the real finetune does (weights come from the pretrained
checkpoint), so any loss difference is attributable to the feature under test and not to
RNG ordering. Prints state_dict key/shape digests, parameter counts, losses, the
regularizer's domain-label counts, and runs a backward pass.

Usage:  python probe.py --stage before      # run BEFORE the patch (writes /tmp/exp0010_before.json)
        python probe.py --stage after       # run AFTER  the patch (compares against it)
"""
import argparse
import hashlib
import json

import torch

from gr00t.model.action_latent_tokenizer_v4_multiemb import MultiEmbActionLatentTokenizerV4

NAME = "openarm_prq"
SPECS = [{"name": NAME, "action_dim": 15, "action_horizon": 16, "class_token_id": 5}]
# Small but structurally faithful: same token_dim/horizon/action_dim as the real run,
# reduced fusion so a CPU forward+backward is seconds not minutes.
SMALL = dict(
    embodiment_specs=SPECS, action_horizon=16, emb_dim=256, head_dim=64,
    encoder_depth=4, decoder_depth=4, decoder_mode="self_attention", token_dim=64,
    dino_dim=64, fusion_width=256, fusion_depth=2, fusion_heads=4, dino_decoder_depth=2,
    use_vae=True, vae_sample=False, lambda_kl=1e-6, lambda_recon=1.0, lambda_dino=0.1,
    recon_loss_type="l1", dino_loss_type="mse",
    dino_loss_weights={"l1": 0.0, "mse": 1.0, "cosine": 1.0},
    dino_final_norm="naive", use_embodiment_class_token=True,
    tokenizer_finetuning_mode=True, new_class_token=1, num_pretrain_class_tokens=5,
)
# The real run's shapes, for a key/shape digest that reflects the actual checkpoint.
REAL = dict(SMALL, dino_dim=1024, fusion_width=1024, fusion_depth=6, dino_decoder_depth=6)

B, T, D, LP = 8, 16, 15, 17
N_HUMAN = 5  # 5 human + 3 robot, so the label counts are unmistakable in the log


def digest(model):
    items = sorted((k, tuple(v.shape)) for k, v in model.state_dict().items())
    return hashlib.sha256(repr(items).encode()).hexdigest()[:16], len(items)


def batch(dino_dim, with_labels):
    torch.manual_seed(1234)
    g = {
        "action": torch.randn(B, T, D),
        "x0_feat": torch.randn(B, LP, dino_dim),
        "x1_feat": torch.randn(B, LP, dino_dim),
    }
    if with_labels:
        lbl = torch.zeros(B)
        lbl[:N_HUMAN] = 1.0  # first N_HUMAN rows are human
        g["is_human"] = lbl
    return {"embodiment_order": [NAME], "groups": {NAME: g}}


def build(**over):
    torch.manual_seed(0)
    return MultiEmbActionLatentTokenizerV4(**{**SMALL, **over})


def run(tag, model, ref_sd, with_labels):
    if ref_sd is not None:
        missing, unexpected = model.load_state_dict(ref_sd, strict=False)
        # Mirror the real finetune's copy-init of the human decoder twin.
        for nm in [k for k in model.recon_decoders if not k.endswith("__human")]:
            if nm + "__human" in model.recon_decoders:
                model.recon_decoders[nm + "__human"].load_state_dict(
                    model.recon_decoders[nm].state_dict())
        newkeys = sorted(missing)
    else:
        newkeys = []
    dg, nkeys = digest(model)
    n_par = sum(p.numel() for p in model.parameters())
    out = model(batch(SMALL["dino_dim"], with_labels))
    out["loss"].backward()
    grads = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    # The groups path tags per-embodiment components as "<key>/<name>"; the regularizer's
    # own diagnostics are global and stay untagged. Strip the tag for comparison.
    scal = {k.split("/")[0]: round(float(v), 10) for k, v in out.items() if v.numel() == 1}
    print(f"\n--- {tag} ---")
    print(f"  sd: {nkeys} keys digest={dg}  params={n_par:,}")
    if newkeys:
        print(f"  keys NOT in the off-baseline ({len(newkeys)}): "
              f"{newkeys[:3]}{' ...' if len(newkeys) > 3 else ''}")
    print(f"  loss={scal['loss']:.10f} recon={scal['loss_recon']:.10f} "
          f"dino={scal['loss_dino']:.10f}")
    for k in ("loss_embod_reg", "embod_reg_n_human", "embod_reg_n_robot",
              "embod_reg_gap", "embod_reg_std_min", "embod_reg_bins"):
        if k in scal:
            print(f"  {k} = {scal[k]}")
    print(f"  backward OK, {grads} param tensors received a nonzero grad")
    return {"digest": dg, "n_keys": nkeys, "n_params": n_par, "scalars": scal,
            "new_keys": newkeys}


ap = argparse.ArgumentParser()
ap.add_argument("--stage", choices=["before", "after"], required=True)
args = ap.parse_args()
REF = "/tmp/exp0010_before.json"

res = {}
off = build()
ref_sd = {k: v.clone() for k, v in off.state_dict().items()}
res["off"] = run("off (baseline)", off, None, with_labels=False)

# Real-shape key/shape digest (no forward) -- what the actual checkpoint will look like.
torch.manual_seed(0)
real = MultiEmbActionLatentTokenizerV4(**REAL)
rd, rn = digest(real)
print(f"\n[real shapes] off: {rn} keys digest={rd} "
      f"params={sum(p.numel() for p in real.parameters()):,}")
res["off_real_digest"] = rd
res["off_real_keys"] = rn
del real

if args.stage == "after":
    res["A"] = run("A: embod_reg vicreg pool=mean",
                   build(embod_reg_mode="vicreg", embod_reg_weight=0.1),
                   ref_sd, with_labels=True)
    res["A_tokens"] = run("A: embod_reg vicreg pool=tokens (time-token bins)",
                          build(embod_reg_mode="vicreg", embod_reg_weight=0.1,
                                embod_reg_pool="tokens"),
                          ref_sd, with_labels=True)
    res["A_dann"] = run("A: embod_reg dann (adds classifier params)",
                        build(embod_reg_mode="dann", embod_reg_weight=0.1),
                        ref_sd, with_labels=True)
    res["B"] = run("B: split recon decoder (copy-init)",
                   build(split_recon_decoder=True), ref_sd, with_labels=True)
    res["AB"] = run("AB: both",
                    build(embod_reg_mode="vicreg", embod_reg_weight=0.1,
                          split_recon_decoder=True), ref_sd, with_labels=True)

    torch.manual_seed(0)
    real_ab = MultiEmbActionLatentTokenizerV4(**dict(
        REAL, embod_reg_mode="vicreg", embod_reg_weight=0.1, split_recon_decoder=True))
    rad, ran = digest(real_ab)
    print(f"\n[real shapes] AB: {ran} keys digest={rad} "
          f"params={sum(p.numel() for p in real_ab.parameters()):,} "
          f"(+{ran - rn} keys vs off)")
    del real_ab

    with open(REF) as f:
        before = json.load(f)
    print("\n===== byte-identity check: default-off vs pre-patch code =====")
    checks = [
        ("state_dict digest (small)", before["off"]["digest"], res["off"]["digest"]),
        ("state_dict key count", before["off"]["n_keys"], res["off"]["n_keys"]),
        ("param count", before["off"]["n_params"], res["off"]["n_params"]),
        ("state_dict digest (real shapes)", before["off_real_digest"], res["off_real_digest"]),
        ("fixed-seed losses", before["off"]["scalars"], res["off"]["scalars"]),
    ]
    ok = True
    for label, a, b in checks:
        good = a == b
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {label}: {a if not good else 'identical'}"
              + (f"  !=  {b}" if not good else ""))
    print("\n===== feature-level checks =====")
    fc = [
        ("B loss == off loss (copy-init split is a no-op at step 0)",
         res["off"]["scalars"]["loss"], res["B"]["scalars"]["loss"]),
        ("A recon == off recon (regularizer does not touch recon)",
         res["off"]["scalars"]["loss_recon"], res["A"]["scalars"]["loss_recon"]),
    ]
    for label, a, b in fc:
        good = a == b
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {label}"
              + ("" if good else f": {a} != {b}"))
    lbl_ok = (res["A"]["scalars"]["embod_reg_n_human"] == N_HUMAN
              and res["A"]["scalars"]["embod_reg_n_robot"] == B - N_HUMAN)
    ok &= lbl_ok
    print(f"  [{'PASS' if lbl_ok else 'FAIL'}] domain labels reached the regularizer: "
          f"human={res['A']['scalars']['embod_reg_n_human']} (expected {N_HUMAN}), "
          f"robot={res['A']['scalars']['embod_reg_n_robot']} (expected {B - N_HUMAN})")
    tok = res["A_tokens"]["scalars"]
    tok_ok = (tok["embod_reg_n_human"] == N_HUMAN * T
              and tok.get("embod_reg_bins", 0) == T)
    ok &= tok_ok
    print(f"  [{'PASS' if tok_ok else 'FAIL'}] pool=tokens stratification: "
          f"human={tok['embod_reg_n_human']} (expected {N_HUMAN * T}), "
          f"bins used={tok.get('embod_reg_bins')} (expected {T})")
    print(f"\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
else:
    with open(REF, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nbaseline written to {REF}")
