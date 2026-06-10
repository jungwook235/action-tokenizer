"""CPU-only smoke test for the V4 tokenizer (no GPU, no DINO model load).

Validates: module imports, forward shapes/finiteness, dino_loss_type switching,
and wrapper round-trip (build-from-state_dict, encode with feats, decode_latent).
Run: CUDA_VISIBLE_DEVICES="" python scripts/_v4_cpu_smoketest.py
"""

import tempfile

import torch

from gr00t.model.action_latent_tokenizer_v4 import (
    ActionLatentTokenizerV4,
    ReconDecoderV4,
    TimeWiseEncoderV4,
)
from gr00t.model.rla_modules import SimpleTokenTransformer
from gr00t.model.action_latent_tokenizer_wrapper import ActionLatentTokenizerWrapper

torch.manual_seed(0)

B, T, D = 2, 16, 29
DINO, LP, TOK, EMB = 1024, 196, 64, 256
FUSION_DEPTH, DEC_DEPTH, ENC_DEPTH, DINO_DEC_DEPTH = 2, 2, 2, 2


def build_model(dino_loss_type="l1"):
    enc = TimeWiseEncoderV4(
        action_dim=D, action_horizon=T, emb_dim=EMB, head_dim=64,
        encoder_depth=ENC_DEPTH, pdropout=0.0, num_global_tokens=0, num_hand_tokens=0,
        dino_dim=DINO, fusion_width=DINO, fusion_depth=FUSION_DEPTH, fusion_heads=16,
        token_dim=TOK,
    )
    dec = ReconDecoderV4(
        action_dim=D, action_horizon=T, emb_dim=EMB, head_dim=64, depth=DEC_DEPTH,
        pdropout=0.0, decoder_mode="self_attention", num_global_tokens=0,
        num_hand_tokens=0, token_dim=TOK,
    )
    dino_dec = SimpleTokenTransformer(
        in_channels=DINO, model_channels=DINO, out_channels=DINO, num_blocks=DINO_DEC_DEPTH,
        num_heads=16, num_tokens=T, token_channels=TOK, zero_init=True, use_fp16=False,
    )
    return ActionLatentTokenizerV4(
        encoder=enc, recon_decoder=dec, dino_decoder=dino_dec,
        lambda_recon=1.0, lambda_dino=1.0, recon_loss_type="mse", dino_loss_type=dino_loss_type,
    )


def main():
    assert not torch.cuda.is_available() or True  # CPU run regardless
    model = build_model("l1").eval()

    actions = torch.randn(B, T, D)
    x0 = torch.randn(B, LP, DINO)
    x1 = torch.randn(B, LP, DINO)

    # ---- forward ----
    out = model({"action": actions, "x0_feat": x0, "x1_feat": x1})
    assert out["loss"].isfinite(), out
    assert out["loss_recon"].isfinite() and out["loss_dino"].isfinite()
    print("[forward] loss=%.4f recon=%.4f dino=%.4f sub=%s"
          % (out["loss"].item(), out["loss_recon"].item(), out["loss_dino"].item(),
             {k: round(v.item(), 4) for k, v in out.items() if k.startswith("loss_dino_")}))

    # ---- encode/decode shapes ----
    g, t, h = model.encode(actions, x0, x1)
    assert t.shape == (B, T, TOK), t.shape
    assert g.shape == (B, 0, TOK) and h.shape == (B, 0, TOK)
    recon = model.decode(g, t, h)
    assert recon.shape == (B, T, D), recon.shape
    pred_x1 = model.decode_dino(t, x0)
    assert pred_x1.shape == (B, LP, DINO), pred_x1.shape
    print("[shapes] latent t=%s recon=%s pred_x1=%s OK" % (tuple(t.shape), tuple(recon.shape), tuple(pred_x1.shape)))

    # ---- dino_loss_type switching ----
    for dlt, expect in [("l1", {"loss_dino_l1"}),
                        ("cosine", {"loss_dino_cosine"}),
                        ("l1+cosine", {"loss_dino_l1", "loss_dino_cosine"})]:
        m = build_model(dlt).eval()
        o = m({"action": actions, "x0_feat": x0, "x1_feat": x1})
        got = {k for k in o if k.startswith("loss_dino_")}
        assert got == expect, f"{dlt}: got {got} expected {expect}"
    print("[dino_loss_type] l1 / cosine / l1+cosine sub-terms OK")

    # ---- wrapper round-trip ----
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ckpt_path = f.name
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

    wrapper = ActionLatentTokenizerWrapper.from_checkpoint(ckpt_path, device="cpu")
    assert wrapper.emb_dim == TOK, wrapper.emb_dim
    assert wrapper.internal_emb_dim == EMB, wrapper.internal_emb_dim
    assert wrapper.get_num_tokens("all") == T, wrapper.get_num_tokens("all")

    latent = wrapper.get_latent_target(actions, target_tokens="all", x0_feat=x0, x1_feat=x1)
    assert latent.shape == (B, T, TOK), latent.shape
    dec_actions = wrapper.decode_latent(latent, target_tokens="all")
    assert dec_actions.shape == (B, T, D), dec_actions.shape

    # consistency vs direct model encode/decode
    gw, tw, hw = wrapper.encode(actions, x0_feat=x0, x1_feat=x1)
    assert torch.allclose(tw, t, atol=1e-4), (tw - t).abs().max()
    assert torch.allclose(dec_actions, recon, atol=1e-4), (dec_actions - recon).abs().max()
    print("[wrapper] _is_v4 load OK; emb_dim=%d num_tokens=%d; encode/decode match model"
          % (wrapper.emb_dim, wrapper.get_num_tokens("all")))

    print("\nALL CPU SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
