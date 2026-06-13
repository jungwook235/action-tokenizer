"""CPU smoke test for the V5 RLA-LAM tokenizer (no weight download).

Covers:
  A. V5 markers + LAM descriptor round-trip — the tokenizer records ``_is_v5`` and
     ``_lam_*`` buffers; the wrapper's ``_build_from_state_dict`` dispatches to the
     v5 builder and decodes them back; the pixel decoder is omitted at rebuild and
     its keys are filtered on strict load.
  B. Forward / shape trace — encode(z_rep) → time latents; decode → actions;
     decode_pixel → future-frame pixels [B,1,H,W,3]; learnable softmax pool.
  C. VAE variant — use_vae records ``_is_vae`` and forward returns loss_kl; wrapper
     rebuild detects it.
  D. V4 regression — a tiny v4 tokenizer is unaffected (no ``_is_v5``, still ``_is_v4``).
"""

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

A, T, E, TD, L = 8, 4, 32, 8, 16  # action_dim, horizon, emb, token_dim, lam_latent
PS, MD, IMG = 4, 16, 8            # pixel decoder: patch_size, model_dim, image HxW
FAKE_CKPT = "/tmp/fake_lam_v5.ckpt"


def build_tiny_v5(use_vae=False):
    from gr00t.model.action_latent_tokenizer_v4 import ReconDecoderV4, TimeWiseEncoderV4
    from gr00t.model.action_latent_tokenizer_v5 import (
        ActionLatentTokenizerV5,
        PixelDecoderV5,
    )

    enc = TimeWiseEncoderV4(
        action_dim=A, action_horizon=T, emb_dim=E, head_dim=16, encoder_depth=1,
        dino_dim=L, fusion_width=E, fusion_depth=1, fusion_heads=2, token_dim=TD,
        use_vae=use_vae,
    )
    dec = ReconDecoderV4(
        action_dim=A, action_horizon=T, emb_dim=E, head_dim=16, depth=1, pdropout=0.0,
        token_dim=TD,
    )
    pix = PixelDecoderV5(
        action_horizon=T, token_dim=TD, image_channels=3, patch_size=PS,
        model_dim=MD, dec_blocks=1, num_heads=2,
    )
    return ActionLatentTokenizerV5(
        encoder=enc, recon_decoder=dec, pixel_decoder=pix,
        lambda_recon=1.0, lambda_pixel=1.0, lambda_kl=1e-6,
        lam_ckpt=FAKE_CKPT, lam_model_dim=MD, lam_latent_dim=L, lam_patch_size=PS,
        lam_enc_blocks=1, lam_dec_blocks=1, lam_num_heads=2, lam_image_h=IMG, lam_image_w=IMG,
    )


def _fake_batch(B=2):
    return {
        "action": torch.randn(B, T, A),
        "z_rep": torch.randn(B, 1, L),
        "frame0": torch.rand(B, 1, IMG, IMG, 3),
        "frame1": torch.rand(B, 1, IMG, IMG, 3),
    }


def test_forward_and_shapes():
    tok = build_tiny_v5()
    B = 2
    batch = _fake_batch(B)
    out = tok(batch)
    assert torch.isfinite(out["loss"]), out
    assert "loss_recon" in out and "loss_pixel" in out

    g, t, h = tok.encode(batch["action"], batch["z_rep"])
    assert t.shape == (B, T, TD), t.shape
    preds = tok.decode(g, t, h)
    assert preds.shape == (B, T, A), preds.shape
    f1hat = tok.decode_pixel(t, batch["frame0"])
    assert f1hat.shape == (B, 1, IMG, IMG, 3), f1hat.shape

    # learnable softmax pool: weights sum to 1, output [B, token_dim]
    w = torch.softmax(tok.pixel_decoder.pool_logits, dim=0)
    assert abs(w.sum().item() - 1.0) < 1e-5
    z_pix = tok.pixel_decoder.pool(t)
    assert z_pix.shape == (B, TD), z_pix.shape
    print(f"[B] forward/shape OK — time {tuple(t.shape)}, actions {tuple(preds.shape)}, "
          f"frame1_hat {tuple(f1hat.shape)}, pool→{tuple(z_pix.shape)}")


def test_markers_roundtrip():
    from gr00t.model.action_latent_tokenizer_wrapper import ActionLatentTokenizerWrapper
    from gr00t.model.action_latent_tokenizer_v5 import (
        ActionLatentTokenizerV5,
        byte_tensor_to_str,
    )

    tok = build_tiny_v5()
    sd = tok.state_dict()
    assert "_is_v5" in sd and "_lam_ckpt" in sd and "_lam_latent_dim" in sd

    rebuilt = ActionLatentTokenizerWrapper._build_from_state_dict(sd)
    assert isinstance(rebuilt, ActionLatentTokenizerV5)
    keys = set(rebuilt.state_dict().keys())
    # pixel decoder is training-only → omitted at rebuild, its keys filtered.
    assert not any(k.startswith("pixel_decoder.") for k in keys), "pixel_decoder leaked into inference model"
    assert any(k.startswith("pixel_decoder.") for k in sd), "tiny v5 should have a pixel_decoder"

    filtered = {k: v for k, v in sd.items() if k in keys}
    rebuilt.load_state_dict(filtered, strict=True)  # mirrors from_checkpoint
    assert byte_tensor_to_str(rebuilt._lam_ckpt) == FAKE_CKPT
    assert int(rebuilt._lam_latent_dim.item()) == L
    assert int(rebuilt._lam_patch_size.item()) == PS
    assert rebuilt.encoder.dino_dim == L
    print(f"[A] V5 markers round-trip OK — wrapper rebuilt v5, dino_dim(lam_latent)={L}, "
          f"pixel_decoder filtered, lam_ckpt round-tripped")


def test_vae_variant():
    from gr00t.model.action_latent_tokenizer_wrapper import ActionLatentTokenizerWrapper

    tok = build_tiny_v5(use_vae=True)
    sd = tok.state_dict()
    assert "_is_vae" in sd
    out = tok(_fake_batch())
    assert "loss_kl" in out and torch.isfinite(out["loss_kl"]), out
    rebuilt = ActionLatentTokenizerWrapper._build_from_state_dict(sd)
    assert rebuilt.use_vae and hasattr(rebuilt.encoder, "logvar_head")
    keys = set(rebuilt.state_dict().keys())
    rebuilt.load_state_dict({k: v for k, v in sd.items() if k in keys}, strict=True)
    print("[C] VAE variant OK — _is_vae recorded, loss_kl returned, wrapper rebuilt VAE encoder")


def test_v4_regression():
    """V5 additions must not perturb V4: a tiny v4 has no _is_v5 and keeps _is_v4."""
    from gr00t.model.action_latent_tokenizer_v4 import (
        ActionLatentTokenizerV4,
        ReconDecoderV4,
        TimeWiseEncoderV4,
    )
    from gr00t.model.rla_modules import SimpleTokenTransformer

    DD = 32
    enc = TimeWiseEncoderV4(
        action_dim=A, action_horizon=T, emb_dim=E, head_dim=16, encoder_depth=1,
        dino_dim=DD, fusion_width=E, fusion_depth=1, fusion_heads=2, token_dim=TD,
    )
    dec = ReconDecoderV4(
        action_dim=A, action_horizon=T, emb_dim=E, head_dim=16, depth=1, pdropout=0.0,
        token_dim=TD,
    )
    dino_dec = SimpleTokenTransformer(
        in_channels=DD, model_channels=E, out_channels=DD, num_blocks=1, num_heads=2,
        num_tokens=T, token_channels=TD, zero_init=True,
    )
    tok = ActionLatentTokenizerV4(encoder=enc, recon_decoder=dec, dino_decoder=dino_dec)
    sd = tok.state_dict()
    assert "_is_v4" in sd and "_is_v5" not in sd
    print("[D] V4 regression OK — v4 tokenizer unaffected (_is_v4 present, no _is_v5)")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_forward_and_shapes()
    test_markers_roundtrip()
    test_vae_variant()
    test_v4_regression()
    print("\nALL V5 SMOKE TESTS PASSED")
