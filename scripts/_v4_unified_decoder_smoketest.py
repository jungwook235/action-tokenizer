"""CPU verification for the V4 unified decoder (--decoder-arch shared_trunk/mot/
shared_trunk_vis).

Checks (run on the head node / a debug pod, small dims, no GPU):
  1. separate (default) is byte-identical to the pre-change code: same state_dict
     keys/shapes/values under the same seed, and EXACTLY equal fixed-seed loss.
     The reference module is loaded from a pristine copy of the previous file
     (pass --orig <path>).
  1c. the three PRE-EXISTING archs (shared_trunk, mot, shared_trunk+segpix) are
     byte-identical to the reference too — this change must not move them.
  2. shared_trunk / mot: small forward + backward, all grads finite.
  3. decode(latent) is a function of the latent alone: the training-pass action
     recon (combined [L,P] with mask) equals the latent-only decode, and swapping
     x0_feat for random noise leaves the action output bitwise unchanged.
  4. parameter counts at the real GR1 config (printed table).
  5. shared_trunk_vis (EXP-0011): the state_dict carries BOTH the separate
     recon_decoder and the visual-only trunk, has NO recon_branch/action_head,
     forward+backward works, all three losses appear, and every parameter gets a
     gradient (DDP-safe).
  6. shared_trunk_vis action independence: the action output is bitwise unchanged
     when x0_feat is replaced by noise AND when the whole unified decoder's weights
     are randomized; and back-propagating loss_recon alone leaves ZERO gradient
     anywhere in the unified decoder.
  7. shared_trunk_vis checkpoints rebuild through the inference wrapper (the
     unified decoder is filtered as training-only, recon_decoder loads strict).

Usage: python scripts/_v4_unified_decoder_smoketest.py --orig /path/to/v4_orig.py
"""

import argparse
import importlib.util
import sys

import torch

torch.set_num_threads(2)  # head-node friendliness


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


SMALL = dict(action_dim=8, action_horizon=16, emb_dim=64, head_dim=32,
             encoder_depth=2, decoder_depth=2, fusion_width=64, fusion_depth=2,
             fusion_heads=4, token_dim=16, dino_channels=48, Lp=9)


def build_separate(mod, cfg, seed=0):
    from gr00t.model.rla_modules import SimpleTokenTransformer
    torch.manual_seed(seed)
    encoder = mod.TimeWiseEncoderV4(
        action_dim=cfg["action_dim"], action_horizon=cfg["action_horizon"],
        emb_dim=cfg["emb_dim"], head_dim=cfg["head_dim"],
        encoder_depth=cfg["encoder_depth"], pdropout=0.0,
        num_global_tokens=0, num_hand_tokens=0, dino_dim=cfg["dino_channels"],
        fusion_width=cfg["fusion_width"], fusion_depth=cfg["fusion_depth"],
        fusion_heads=cfg["fusion_heads"], token_dim=cfg["token_dim"],
        use_vae=True, vae_sample=True,
    )
    recon_decoder = mod.ReconDecoderV4(
        action_dim=cfg["action_dim"], action_horizon=cfg["action_horizon"],
        emb_dim=cfg["emb_dim"], head_dim=cfg["head_dim"], depth=cfg["decoder_depth"],
        pdropout=0.0, decoder_mode="self_attention",
        num_global_tokens=0, num_hand_tokens=0, token_dim=cfg["token_dim"],
    )
    dino_decoder = SimpleTokenTransformer(
        in_channels=cfg["dino_channels"], model_channels=cfg["fusion_width"],
        out_channels=cfg["dino_channels"], num_blocks=2,
        num_heads=cfg["fusion_heads"], num_tokens=cfg["action_horizon"],
        token_channels=cfg["token_dim"], zero_init=True, use_fp16=False,
    )
    return mod.ActionLatentTokenizerV4(
        encoder=encoder, recon_decoder=recon_decoder, dino_decoder=dino_decoder,
        lambda_recon=1.0, lambda_dino=0.1, lambda_kl=1e-6,
        recon_loss_type="l1", dino_loss_type="mse", dino_final_norm="naive",
    )


def build_unified(mod, cfg, arch, seed=0, recon_sees_vision=False,
                  trunk_depth=2, branch_depth=1, mot_depth=2,
                  use_segpix=False, seg_pixel_patch=4):
    torch.manual_seed(seed)
    encoder = mod.TimeWiseEncoderV4(
        action_dim=cfg["action_dim"], action_horizon=cfg["action_horizon"],
        emb_dim=cfg["emb_dim"], head_dim=cfg["head_dim"],
        encoder_depth=cfg["encoder_depth"], pdropout=0.0,
        num_global_tokens=0, num_hand_tokens=0, dino_dim=cfg["dino_channels"],
        fusion_width=cfg["fusion_width"], fusion_depth=cfg["fusion_depth"],
        fusion_heads=cfg["fusion_heads"], token_dim=cfg["token_dim"],
        use_vae=False,
    )
    ud = mod.UnifiedDecoderV4(
        action_dim=cfg["action_dim"], action_horizon=cfg["action_horizon"],
        token_dim=cfg["token_dim"], dino_dim=cfg["dino_channels"],
        width=cfg["fusion_width"], head_dim=cfg["head_dim"], arch=arch,
        trunk_depth=trunk_depth, branch_depth=branch_depth, mot_depth=mot_depth,
        recon_sees_vision=recon_sees_vision,
        **({"use_segpix_branch": True} if use_segpix else {}),
    )
    seg_pixel_head = (
        mod.LinearHead(cfg["fusion_width"], seg_pixel_patch ** 2, weight_init_style="zero")
        if use_segpix else None
    )
    return mod.ActionLatentTokenizerV4(
        encoder=encoder, recon_decoder=None, dino_decoder=None,
        unified_decoder=ud, decoder_arch=arch,
        seg_pixel_head=seg_pixel_head, seg_pixel_patch=seg_pixel_patch,
        lambda_seg_pixel=0.1 if use_segpix else 0.0,
        lambda_recon=1.0, lambda_dino=0.1,
        recon_loss_type="l1", dino_loss_type="mse", dino_final_norm="naive",
    )


def build_unified_vis(mod, cfg, seed=0, trunk_depth=2, branch_depth=1,
                      use_segpix=False, seg_pixel_patch=4):
    """EXP-0011: visual-only trunk (no recon branch) + the separate ReconDecoderV4.

    Build order mirrors the trainer: encoder → unified decoder → recon decoder →
    segpix head.
    """
    torch.manual_seed(seed)
    encoder = mod.TimeWiseEncoderV4(
        action_dim=cfg["action_dim"], action_horizon=cfg["action_horizon"],
        emb_dim=cfg["emb_dim"], head_dim=cfg["head_dim"],
        encoder_depth=cfg["encoder_depth"], pdropout=0.0,
        num_global_tokens=0, num_hand_tokens=0, dino_dim=cfg["dino_channels"],
        fusion_width=cfg["fusion_width"], fusion_depth=cfg["fusion_depth"],
        fusion_heads=cfg["fusion_heads"], token_dim=cfg["token_dim"],
        use_vae=False,
    )
    ud = mod.UnifiedDecoderV4(
        action_dim=cfg["action_dim"], action_horizon=cfg["action_horizon"],
        token_dim=cfg["token_dim"], dino_dim=cfg["dino_channels"],
        width=cfg["fusion_width"], head_dim=cfg["head_dim"], arch="shared_trunk_vis",
        trunk_depth=trunk_depth, branch_depth=branch_depth,
        use_segpix_branch=use_segpix,
    )
    recon_decoder = mod.ReconDecoderV4(
        action_dim=cfg["action_dim"], action_horizon=cfg["action_horizon"],
        emb_dim=cfg["emb_dim"], head_dim=cfg["head_dim"], depth=cfg["decoder_depth"],
        pdropout=0.0, decoder_mode="self_attention",
        num_global_tokens=0, num_hand_tokens=0, token_dim=cfg["token_dim"],
    )
    seg_pixel_head = (
        mod.LinearHead(cfg["fusion_width"], seg_pixel_patch ** 2, weight_init_style="zero")
        if use_segpix else None
    )
    return mod.ActionLatentTokenizerV4(
        encoder=encoder, recon_decoder=recon_decoder, dino_decoder=None,
        unified_decoder=ud, decoder_arch="shared_trunk_vis",
        seg_pixel_head=seg_pixel_head, seg_pixel_patch=seg_pixel_patch,
        lambda_seg_pixel=0.1 if use_segpix else 0.0,
        lambda_recon=1.0, lambda_dino=0.1,
        recon_loss_type="l1", dino_loss_type="mse", dino_final_norm="naive",
    )


def build_separate_segpix(mod, cfg, seed=0, seg_pixel_patch=4):
    """EXP-0003 path: separate arch + seg_pixel_decoder/head (both modules)."""
    from gr00t.model.rla_modules import SimpleTokenTransformer
    base = build_separate(mod, cfg, seed=seed)  # consumes the same RNG prefix
    seg_pixel_decoder = SimpleTokenTransformer(
        in_channels=cfg["dino_channels"], model_channels=cfg["fusion_width"],
        out_channels=cfg["dino_channels"], num_blocks=2,
        num_heads=cfg["fusion_heads"], num_tokens=cfg["action_horizon"],
        token_channels=cfg["token_dim"], zero_init=True, use_fp16=False,
    )
    seg_pixel_head = mod.LinearHead(
        cfg["dino_channels"], seg_pixel_patch ** 2, weight_init_style="zero"
    )
    return mod.ActionLatentTokenizerV4(
        encoder=base.encoder, recon_decoder=base.recon_decoder,
        dino_decoder=base.dino_decoder,
        seg_pixel_decoder=seg_pixel_decoder, seg_pixel_head=seg_pixel_head,
        seg_pixel_patch=seg_pixel_patch, lambda_seg_pixel=0.1,
        lambda_recon=1.0, lambda_dino=0.1, lambda_kl=1e-6,
        recon_loss_type="l1", dino_loss_type="mse", dino_final_norm="naive",
    )


def make_batch(cfg, seed=7, with_masks=False, seg_pixel_patch=4):
    torch.manual_seed(seed)
    batch = {
        "action": torch.randn(2, cfg["action_horizon"], cfg["action_dim"]),
        "x0_feat": torch.randn(2, cfg["Lp"], cfg["dino_channels"]),
        "x1_feat": torch.randn(2, cfg["Lp"], cfg["dino_channels"]),
    }
    if with_masks:
        grid = int(round(cfg["Lp"] ** 0.5))
        S = grid * seg_pixel_patch
        batch["mask_x1"] = (torch.rand(2, S, S) > 0.5).to(torch.uint8)
        batch["mask_valid"] = torch.tensor([1, 1])
    return batch


def check1_byte_identical(new_mod, orig_mod):
    old = build_separate(orig_mod, SMALL, seed=0)
    new = build_separate(new_mod, SMALL, seed=0)
    sd_o, sd_n = old.state_dict(), new.state_dict()
    assert list(sd_o.keys()) == list(sd_n.keys()), (
        f"state_dict keys differ:\n only-old={set(sd_o) - set(sd_n)}\n"
        f" only-new={set(sd_n) - set(sd_o)}"
    )
    for k in sd_o:
        assert sd_o[k].shape == sd_n[k].shape, f"shape mismatch at {k}"
        assert torch.equal(sd_o[k], sd_n[k]), f"value mismatch at {k}"
    batch = make_batch(SMALL)
    old.train(); new.train()
    torch.manual_seed(123)
    out_o = old(dict(batch))
    torch.manual_seed(123)
    out_n = new(dict(batch))
    for key in ("loss", "loss_recon", "loss_dino", "loss_kl"):
        vo, vn = out_o[key].item(), out_n[key].item()
        assert vo == vn, f"{key} differs: old={vo!r} new={vn!r}"
    print(f"[1] separate byte-identical: {len(sd_o)} keys equal, "
          f"loss == {out_o['loss'].item():.10f} (exact match incl. recon/dino/kl)")


def check1c_unified_archs_identical(new_mod, orig_mod):
    """The three PRE-EXISTING unified configurations must not move at all."""
    for arch, segpix in (("shared_trunk", False), ("mot", False), ("shared_trunk", True)):
        old = build_unified(orig_mod, SMALL, arch, seed=0, use_segpix=segpix)
        new = build_unified(new_mod, SMALL, arch, seed=0, use_segpix=segpix)
        sd_o, sd_n = old.state_dict(), new.state_dict()
        tag = f"{arch}{'+segpix' if segpix else ''}"
        assert list(sd_o.keys()) == list(sd_n.keys()), (
            f"{tag}: state_dict keys differ: only-old={set(sd_o) - set(sd_n)} "
            f"only-new={set(sd_n) - set(sd_o)}"
        )
        for k in sd_o:
            assert sd_o[k].shape == sd_n[k].shape, f"{tag}: shape mismatch at {k}"
            assert torch.equal(sd_o[k], sd_n[k]), f"{tag}: value mismatch at {k}"
        batch = make_batch(SMALL, with_masks=segpix)
        old.train(); new.train()
        torch.manual_seed(77)
        out_o = old(dict(batch))
        torch.manual_seed(77)
        out_n = new(dict(batch))
        assert set(out_o) == set(out_n), f"{tag}: output keys differ"
        for key in out_o:
            vo, vn = out_o[key].item(), out_n[key].item()
            assert vo == vn, f"{tag}: {key} differs: old={vo!r} new={vn!r}"
        print(f"[1c] {tag:22s} byte-identical: {len(sd_o)} keys, "
              f"loss == {out_o['loss'].item():.10f} ({len(out_o)} output terms equal)")


def check5_visual_only_smoke(new_mod, use_segpix):
    model = build_unified_vis(new_mod, SMALL, seed=1, use_segpix=use_segpix)
    sd = model.state_dict()
    tag = f"shared_trunk_vis{'+segpix' if use_segpix else ''}"
    assert any(k.startswith("recon_decoder.") for k in sd), "separate recon decoder missing"
    assert not any(k.startswith("dino_decoder.") for k in sd), "dino_decoder must be gone"
    assert any(k.startswith("unified_decoder.trunk.") for k in sd)
    assert any(k.startswith("unified_decoder.dino_branch.") for k in sd)
    assert not any(k.startswith("unified_decoder.recon_branch.") for k in sd), (
        "recon branch must NOT exist in the visual-only trunk"
    )
    assert not any(k.startswith("unified_decoder.action_head.") for k in sd), (
        "action head must NOT exist in the visual-only trunk"
    )
    assert any(k.startswith("unified_decoder.segpix_branch.") for k in sd) == use_segpix
    assert any(k.startswith("seg_pixel_head.") for k in sd) == use_segpix
    assert "_decoder_arch" in sd
    batch = make_batch(SMALL, with_masks=use_segpix)
    model.train()
    out = model(dict(batch))
    for key in ("loss", "loss_recon", "loss_dino") + (("loss_seg_pixel",) if use_segpix else ()):
        assert key in out and torch.isfinite(out[key]), f"{tag}: bad {key}"
    out["loss"].backward()
    # frozen sincos pos-embs are nn.Parameters with requires_grad=False — DDP
    # ignores them; every TRAINABLE tensor must receive a gradient.
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    missing = [n for n, p in trainable if p.grad is None]
    assert not missing, f"{tag}: trainable params without grad (DDP-unsafe): {missing[:5]}"
    assert all(torch.isfinite(p.grad).all() for _, p in trainable)
    terms = " ".join(f"{k}={out[k].item():.6f}" for k in
                     ("loss_recon", "loss_dino") + (("loss_seg_pixel",) if use_segpix else ()))
    print(f"[5] {tag}: fwd+bwd OK, {terms}; grads on ALL {len(trainable)} trainable "
          f"param tensors (no unused → DDP-safe)")


def check6_action_trunk_independence(new_mod, use_segpix=True):
    """The action path must be provably independent of the unified (visual) trunk."""
    model = build_unified_vis(new_mod, SMALL, seed=2, use_segpix=use_segpix)
    model.eval()
    batch = make_batch(SMALL, with_masks=use_segpix)
    with torch.no_grad():
        g, t, h = model.encode(batch["action"], batch["x0_feat"], batch["x1_feat"])
        act0 = model.decode(g, t, h)
        act_noise = model.decode(g, t, h, x0_feat=torch.randn_like(batch["x0_feat"]) * 100.0)
        _, feat_before, _ = model.unified_decoder(None, t, None, batch["x0_feat"])
        # destroy every weight in the unified decoder
        for p in model.unified_decoder.parameters():
            p.copy_(torch.randn_like(p) * 5.0)
        act_wrecked = model.decode(g, t, h)
        _, feat_after, _ = model.unified_decoder(None, t, None, batch["x0_feat"])
    assert torch.equal(act0, act_noise), "action changed when x0_feat changed"
    assert torch.equal(act0, act_wrecked), (
        "action changed when the unified decoder's weights were randomized — the "
        "action path is contaminated by the trunk"
    )
    assert not torch.equal(feat_before, feat_after), (
        "randomizing the trunk did not change the visual output — the check is vacuous"
    )

    # gradient-level proof: the action loss alone must not reach the trunk at all.
    m2 = build_unified_vis(new_mod, SMALL, seed=2, use_segpix=use_segpix)
    m2.train()
    out = m2(dict(make_batch(SMALL, with_masks=use_segpix)))
    out["loss_recon"].backward()
    leaked = [
        n for n, p in m2.named_parameters()
        if n.startswith("unified_decoder.") and p.grad is not None and p.grad.abs().max() > 0
    ]
    assert not leaked, f"loss_recon leaked gradient into the trunk: {leaked[:5]}"
    n_ud = sum(1 for n, _ in m2.named_parameters() if n.startswith("unified_decoder."))
    print(f"[6] action ⟂ trunk: x0→noise and trunk-weights→random both leave the action "
          f"BITWISE identical (visual output does change); loss_recon.backward() gives "
          f"zero grad on all {n_ud} unified-decoder tensors")


def check7_visual_only_wrapper_rebuild(new_mod, use_segpix=True):
    """Stage-2 path: the inference wrapper must rebuild a shared_trunk_vis ckpt."""
    from gr00t.model.action_latent_tokenizer_wrapper import ActionLatentTokenizerWrapper

    # The wrapper INFERS head_dim (prefers 64) and fusion_heads (= fusion_width//64)
    # from the state_dict, so use dims where those heuristics match the build —
    # otherwise the attention head split differs and the comparison is vacuous.
    cfg = dict(SMALL, emb_dim=128, head_dim=64, fusion_width=128, fusion_heads=2)
    model = build_unified_vis(new_mod, cfg, seed=4, use_segpix=use_segpix)
    sd = model.state_dict()
    rebuilt = ActionLatentTokenizerWrapper._build_from_state_dict(sd)
    keys = set(rebuilt.state_dict().keys())
    assert not any(k.startswith("unified_decoder.") for k in keys), (
        "the wrapper rebuilt the (training-only) visual trunk"
    )
    assert any(k.startswith("recon_decoder.") for k in keys), "no recon decoder at inference"
    filtered = {k: v for k, v in sd.items() if k in keys}
    missing = keys - set(filtered)
    assert not missing, f"wrapper expects keys the checkpoint lacks: {sorted(missing)[:5]}"
    rebuilt.load_state_dict(filtered, strict=True)
    skipped = len(sd) - len(filtered)
    # and the rebuilt decode must reproduce the training-time action exactly
    batch = make_batch(cfg, with_masks=use_segpix)
    model.eval(); rebuilt.eval()
    with torch.no_grad():
        g, t, h = model.encode(batch["action"], batch["x0_feat"], batch["x1_feat"])
        a_train = model.decode(g, t, h)
        g2, t2, h2 = rebuilt.encode(batch["action"], batch["x0_feat"], batch["x1_feat"])
        a_infer = rebuilt.decode(g2, t2, h2)
    assert torch.equal(t, t2), "wrapper-rebuilt encoder produced a different latent"
    assert torch.equal(a_train, a_infer), "wrapper-rebuilt action decode differs"
    print(f"[7] wrapper rebuild OK: strict load on {len(filtered)} keys, "
          f"{skipped} training-only keys filtered, latent + action bitwise identical")


def check2_smoke(new_mod, arch):
    model = build_unified(new_mod, SMALL, arch, seed=1)
    sd = model.state_dict()
    assert not any(k.startswith(("recon_decoder.", "dino_decoder.")) for k in sd)
    assert "_decoder_arch" in sd
    batch = make_batch(SMALL)
    model.train()
    out = model(dict(batch))
    out["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    n_grad = sum(1 for p in model.parameters() if p.grad is not None)
    n_par = sum(1 for _ in model.parameters())
    print(f"[2] {arch}: fwd+bwd OK, loss={out['loss'].item():.6f}, "
          f"grads finite on {n_grad}/{n_par} param tensors")


def check2b_separate_segpix_identical(new_mod, orig_mod):
    """EXP-0003 path (separate + seg_pixel_decoder) is untouched by this change."""
    old = build_separate_segpix(orig_mod, SMALL, seed=0)
    new = build_separate_segpix(new_mod, SMALL, seed=0)
    sd_o, sd_n = old.state_dict(), new.state_dict()
    assert list(sd_o.keys()) == list(sd_n.keys()), (
        f"segpix state_dict keys differ: only-old={set(sd_o) - set(sd_n)} "
        f"only-new={set(sd_n) - set(sd_o)}"
    )
    for k in sd_o:
        assert torch.equal(sd_o[k], sd_n[k]), f"value mismatch at {k}"
    batch = make_batch(SMALL, with_masks=True)
    old.train(); new.train()
    torch.manual_seed(321)
    out_o = old(dict(batch))
    torch.manual_seed(321)
    out_n = new(dict(batch))
    for key in ("loss", "loss_recon", "loss_dino", "loss_kl", "loss_seg_pixel"):
        assert out_o[key].item() == out_n[key].item(), (
            f"{key} differs: old={out_o[key].item()!r} new={out_n[key].item()!r}"
        )
    print(f"[2b] separate+segpix (EXP-0003) byte-identical: {len(sd_o)} keys, "
          f"loss == {out_o['loss'].item():.10f}, "
          f"loss_seg_pixel == {out_o['loss_seg_pixel'].item():.10f}")


def check2c_shared_trunk_segpix_smoke(new_mod):
    model = build_unified(new_mod, SMALL, "shared_trunk", seed=1, use_segpix=True)
    sd = model.state_dict()
    assert any(k.startswith("unified_decoder.segpix_branch.") for k in sd)
    assert any(k.startswith("seg_pixel_head.") for k in sd)
    assert not any(k.startswith("seg_pixel_decoder.") for k in sd)
    batch = make_batch(SMALL, with_masks=True)
    model.train()
    out = model(dict(batch))
    assert "loss_seg_pixel" in out and torch.isfinite(out["loss_seg_pixel"])
    out["loss"].backward()
    gnames = [n for n, p in model.named_parameters() if p.grad is not None]
    assert any(n.startswith("unified_decoder.segpix_branch.") for n in gnames)
    assert any(n.startswith("seg_pixel_head.") for n in gnames)
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert all(torch.isfinite(g).all() for g in grads)
    print(f"[2c] shared_trunk+segpix: fwd+bwd OK, loss={out['loss'].item():.6f}, "
          f"loss_seg_pixel={out['loss_seg_pixel'].item():.6f} (~ln2 at zero-init), "
          f"grads reach segpix branch + head")


def check4b_segpix_forward_independence(new_mod):
    """segpix on/off models share identical params (branch is built LAST), so the
    fixed-seed recon/dino losses must be EXACTLY equal at the forward level."""
    a = build_unified(new_mod, SMALL, "shared_trunk", seed=3, use_segpix=False)
    b = build_unified(new_mod, SMALL, "shared_trunk", seed=3, use_segpix=True)
    a.eval(); b.eval()
    batch_a = make_batch(SMALL)
    batch_b = make_batch(SMALL, with_masks=True)
    with torch.no_grad():
        out_a = a(dict(batch_a))
        out_b = b(dict(batch_b))
    for key in ("loss_recon", "loss_dino"):
        va, vb = out_a[key].item(), out_b[key].item()
        assert va == vb, f"{key} differs with segpix on/off: {va!r} vs {vb!r}"
    assert "loss_seg_pixel" not in out_a and "loss_seg_pixel" in out_b
    print(f"[4b] segpix branch forward-independent: recon {out_a['loss_recon'].item():.10f} "
          f"and dino {out_a['loss_dino'].item():.10f} identical with the branch on/off")


def check3_latent_only(new_mod, arch, use_segpix=False):
    model = build_unified(new_mod, SMALL, arch, seed=2, use_segpix=use_segpix)
    model.eval()
    batch = make_batch(SMALL)
    with torch.no_grad():
        g, t, h = model.encode(batch["action"], batch["x0_feat"], batch["x1_feat"])
        act_joint, _, _ = model.unified_decoder(g, t, h, batch["x0_feat"])
        act_lat_only = model.decode(g, t, h)  # no image
        x0_noise = torch.randn_like(batch["x0_feat"]) * 100.0
        act_noise, _, _ = model.unified_decoder(g, t, h, x0_noise)
    assert torch.equal(act_joint, act_noise), (
        f"{arch}: action recon CHANGED when x0_feat changed — vision leaks into "
        "the latent positions (mask broken)"
    )
    md = (act_joint - act_lat_only).abs().max().item()
    assert torch.allclose(act_joint, act_lat_only, atol=1e-5), (
        f"{arch}: latent-only decode differs from joint pass (max diff {md:.3e})"
    )
    print(f"[3] {arch}: x0 swap → bitwise-identical action (P never enters L); "
          f"latent-only decode max|Δ|={md:.2e}")


def check4_param_table(new_mod, action_dim, action_horizon):
    real = dict(SMALL, action_dim=action_dim, action_horizon=action_horizon,
                emb_dim=256, head_dim=64, encoder_depth=4, decoder_depth=4,
                fusion_width=1024, fusion_depth=6, fusion_heads=16,
                token_dim=64, dino_channels=1024, Lp=256)
    def count(m):
        return sum(p.numel() for p in m.parameters())
    sep = build_separate(new_mod, dict(real, decoder_depth=4), seed=0)
    # ref recipe: dino_decoder depth 6 (build_separate hardcodes 2 for speed) —
    # rebuild the dino decoder at depth 6 for a faithful count
    from gr00t.model.rla_modules import SimpleTokenTransformer
    torch.manual_seed(0)
    dd6 = SimpleTokenTransformer(
        in_channels=1024, model_channels=1024, out_channels=1024, num_blocks=6,
        num_heads=16, num_tokens=action_horizon, token_channels=64,
        zero_init=True, use_fp16=False)
    sep_total = count(sep) - count(sep.dino_decoder) + count(dd6)
    st = build_unified(new_mod, real, "shared_trunk", seed=0,
                       trunk_depth=4, branch_depth=2)
    stp = build_unified(new_mod, real, "shared_trunk", seed=0,
                        trunk_depth=4, branch_depth=2,
                        use_segpix=True, seg_pixel_patch=14)
    mot = build_unified(new_mod, real, "mot", seed=0, mot_depth=6)
    stv = build_unified_vis(new_mod, real, seed=0, trunk_depth=4, branch_depth=2,
                            use_segpix=True, seg_pixel_patch=14)
    rows = [
        ("separate (enc + recon d4 + dino d6)", sep_total,
         count(sep.recon_decoder) + count(dd6)),
        ("shared_trunk (trunk4 + 2x branch2)", count(st), count(st.unified_decoder)),
        ("shared_trunk + segpix branch2 + head", count(stp),
         count(stp.unified_decoder) + count(stp.seg_pixel_head)),
        ("mot (6 layers, per-group FFN)", count(mot), count(mot.unified_decoder)),
        ("shared_trunk_vis: trunk4 + dino/segpix b2 + sep recon d4", count(stv),
         count(stv.unified_decoder) + count(stv.seg_pixel_head) + count(stv.recon_decoder)),
    ]
    print(f"[4] parameter counts @ action_dim={action_dim}, horizon={action_horizon}:")
    for name, total, dec in rows:
        print(f"    {name:58s} total={total/1e6:7.2f}M  decoder-side={dec/1e6:7.2f}M")
    vis_dec = (count(stv.unified_decoder) + count(stv.seg_pixel_head)
               + count(stv.recon_decoder))
    st_dec = count(stp.unified_decoder) + count(stp.seg_pixel_head)
    print(f"    → shared_trunk_vis decoder-side {vis_dec/1e6:.2f}M vs current "
          f"st-segpix {st_dec/1e6:.2f}M (Δ {-(st_dec - vis_dec)/1e6:+.2f}M)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True, help="pristine f2b84ea copy of action_latent_tokenizer_v4.py")
    ap.add_argument("--action-dim", type=int, default=29)
    ap.add_argument("--action-horizon", type=int, default=16)
    args = ap.parse_args()

    import gr00t.model.action_latent_tokenizer_v4 as new_mod
    orig_mod = load_module(args.orig, "v4_orig_f2b84ea")

    check1_byte_identical(new_mod, orig_mod)
    check2b_separate_segpix_identical(new_mod, orig_mod)
    check1c_unified_archs_identical(new_mod, orig_mod)
    for arch in ("shared_trunk", "mot"):
        check2_smoke(new_mod, arch)
        check3_latent_only(new_mod, arch)
    check2c_shared_trunk_segpix_smoke(new_mod)
    check4b_segpix_forward_independence(new_mod)
    check3_latent_only(new_mod, "shared_trunk", use_segpix=True)  # no leak into L
    # EXP-0011: visual-only trunk
    check5_visual_only_smoke(new_mod, use_segpix=False)
    check5_visual_only_smoke(new_mod, use_segpix=True)
    check6_action_trunk_independence(new_mod, use_segpix=True)
    check7_visual_only_wrapper_rebuild(new_mod, use_segpix=True)
    check4_param_table(new_mod, args.action_dim, args.action_horizon)
    print("ALL CHECKS PASSED")
