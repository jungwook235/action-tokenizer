"""CPU smoke test for the V4 DINO↔VGGT feature-source toggle (no weight download).

Covers:
  A. DINO regression — feature_source="dino" registers NO new buffers (state_dict
     byte-identical to before); old DINO checkpoints still load strict.
  B. VGGT markers round-trip — feature_source="vggt" records buffers; the wrapper's
     _build_from_state_dict decodes them back.
  C. Extraction math — the dpt_out2 reproduction (real DPTHead + synthetic tokens)
     and the aggregator slice produce [B,256,1024] / [B,256,2048].
  D. Aggregator structure — a tiny conv-patch Aggregator returns the cached-layer
     list + patch_start_idx that the extractor's slicing assumes.
"""

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "vggt"))


def build_tiny_v4(feature_source, **vggt_kw):
    from gr00t.model.action_latent_tokenizer_v4 import (
        ActionLatentTokenizerV4,
        ReconDecoderV4,
        TimeWiseEncoderV4,
    )
    from gr00t.model.rla_modules import SimpleTokenTransformer

    A, T, E, TD, DD = 8, 4, 32, 8, 1024
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
    return ActionLatentTokenizerV4(
        encoder=enc, recon_decoder=dec, dino_decoder=dino_dec,
        feature_source=feature_source, **vggt_kw,
    )


def test_dino_regression():
    tok = build_tiny_v4("dino")
    sd = tok.state_dict()
    bad = [k for k in sd if k.startswith("_feature_source") or k.startswith("_vggt")
           or k.startswith("_dino_final_norm")]
    assert not bad, f"DINO (affine) tokenizer leaked markers: {bad}"
    assert tok.feature_source == "dino" and tok.dino_final_norm == "affine"
    print("[A] DINO regression OK — no markers, state_dict keys:", len(sd))


def test_dino_naive_ln_marker():
    from gr00t.model.action_latent_tokenizer_wrapper import ActionLatentTokenizerWrapper
    from gr00t.model.action_latent_tokenizer_v4 import byte_tensor_to_str

    tok = build_tiny_v4("dino", dino_final_norm="naive")
    sd = tok.state_dict()
    assert "_dino_final_norm" in sd and "_feature_source" not in sd
    rebuilt = ActionLatentTokenizerWrapper._build_from_state_dict(sd)
    keys = set(rebuilt.state_dict().keys())
    rebuilt.load_state_dict({k: v for k, v in sd.items() if k in keys}, strict=True)
    assert rebuilt.feature_source == "dino"
    assert byte_tensor_to_str(rebuilt._dino_final_norm) == "naive"
    assert rebuilt.dino_final_norm == "naive"
    print("[A2] DINO naive-LN marker round-trip OK — wrapper detected dino/naive")


def test_vggt_markers_roundtrip():
    from gr00t.model.action_latent_tokenizer_wrapper import ActionLatentTokenizerWrapper

    tok = build_tiny_v4(
        "vggt", vggt_token_source="dpt_out2", vggt_image_size=224,
        vggt_model="facebook/VGGT-1B",
    )
    sd = tok.state_dict()
    assert "_feature_source" in sd and "_vggt_token_source" in sd
    rebuilt = ActionLatentTokenizerWrapper._build_from_state_dict(sd)
    # filtered strict load (mirrors from_checkpoint)
    keys = set(rebuilt.state_dict().keys())
    rebuilt.load_state_dict({k: v for k, v in sd.items() if k in keys}, strict=True)
    assert rebuilt.feature_source == "vggt"
    from gr00t.model.action_latent_tokenizer_v4 import byte_tensor_to_str
    assert byte_tensor_to_str(rebuilt._vggt_token_source) == "dpt_out2"
    assert int(rebuilt._vggt_image_size.item()) == 224
    print("[B] VGGT markers round-trip OK — wrapper detected vggt/dpt_out2/224")


def test_extraction_math():
    from vggt.heads.dpt_head import DPTHead

    B, ps, img = 2, 14, 224
    Lp = (img // ps) ** 2  # 256
    patch_start_idx = 5
    P = patch_start_idx + Lp
    C = 2048

    head = DPTHead(dim_in=C, output_dim=4, activation="inv_log", conf_activation="expp1")
    head.eval()

    # Synthetic aggregated_tokens_list: only cached layers populated.
    tokens_list = [None] * 24
    for li in (4, 11, 17, 23):
        tokens_list[li] = torch.randn(B, 1, P, C)

    # --- dpt_out2 (copy of VGGTFeatureExtractor._dpt_out2) ---
    dpt_idx = 2
    layer_idx = head.intermediate_layer_idx[dpt_idx]
    ph = pw = img // ps
    x = tokens_list[layer_idx][:, :, patch_start_idx:]
    x = x.reshape(B, -1, x.shape[-1])
    x = head.norm(x)
    x = x.permute(0, 2, 1).reshape(B, x.shape[-1], ph, pw)
    x = head.projects[dpt_idx](x)
    if head.pos_embed:
        x = head._apply_pos_embed(x, img, img)
    x = head.resize_layers[dpt_idx](x)
    tok_dpt = x.flatten(2).transpose(1, 2)
    assert tok_dpt.shape == (B, Lp, 1024), tok_dpt.shape

    # --- aggregator slice ---
    tok_agg = tokens_list[-1][:, 0, patch_start_idx:, :]
    assert tok_agg.shape == (B, Lp, 2048), tok_agg.shape
    print(f"[C] extraction math OK — dpt_out2 {tuple(tok_dpt.shape)}, aggregator {tuple(tok_agg.shape)}")


def test_tiny_aggregator_structure():
    from vggt.models.aggregator import Aggregator

    B, img, ps, E = 2, 28, 14, 64  # tiny: 28/14 = 2 -> 4 patches
    agg = Aggregator(img_size=img, patch_size=ps, embed_dim=E, depth=2, num_heads=2,
                     patch_embed="conv", num_register_tokens=4)
    agg.eval()
    with torch.no_grad():
        tokens_list, psi = agg(torch.rand(B, 1, 3, img, img))
    assert psi == 5, psi  # 1 camera + 4 register
    last = tokens_list[-1]
    assert last is not None and last.shape[-1] == 2 * E, last.shape  # frame+global concat
    Lp = (img // ps) ** 2
    patch = last[:, 0, psi:, :]
    assert patch.shape == (B, Lp, 2 * E), patch.shape
    print(f"[D] tiny aggregator OK — patch_start_idx={psi}, last layer patch {tuple(patch.shape)}")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_dino_regression()
    test_dino_naive_ln_marker()
    test_vggt_markers_roundtrip()
    test_extraction_math()
    test_tiny_aggregator_structure()
    print("\nALL SMOKE TESTS PASSED")
