"""CPU-only Phase-2 (VLA integration) smoke test — no GPU, no DINO download.

Verifies:
  - imports of the V4 VLA dataset, modified VLA model, trainer, finetune script,
  - wrapper.get_latent_target(actions, x0=frames, x1=frames) with a FAKE DINO
    extractor injected (so no network / GPU), returns the expected latent shape.
Run: CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python scripts/_v4_phase2_cpu_smoketest.py
"""

import importlib
import tempfile

import torch

from gr00t.model.action_latent_tokenizer_v4 import (
    ActionLatentTokenizerV4,
    ReconDecoderV4,
    TimeWiseEncoderV4,
)
from gr00t.model.rla_modules import SimpleTokenTransformer
from gr00t.model.action_latent_tokenizer_wrapper import ActionLatentTokenizerWrapper

B, T, D = 2, 16, 29
DINO, LP, TOK, EMB = 1024, 196, 64, 256


class _FakeDino:
    """Stand-in for DINOv3FeatureExtractor: [B,3,H,W] → (cls, grid[B,C,h,w])."""

    def __init__(self, dim=DINO, grid=14):
        self.dim, self.grid = dim, grid

    def __call__(self, frames, return_spatial_grid=True):
        b = frames.shape[0]
        return None, torch.randn(b, self.dim, self.grid, self.grid)


def build_and_save():
    enc = TimeWiseEncoderV4(
        action_dim=D, action_horizon=T, emb_dim=EMB, head_dim=64, encoder_depth=2,
        pdropout=0.0, dino_dim=DINO, fusion_width=DINO, fusion_depth=2, fusion_heads=16,
        token_dim=TOK,
    )
    dec = ReconDecoderV4(
        action_dim=D, action_horizon=T, emb_dim=EMB, head_dim=64, depth=2, pdropout=0.0,
        decoder_mode="self_attention", token_dim=TOK,
    )
    dino_dec = SimpleTokenTransformer(
        in_channels=DINO, model_channels=DINO, out_channels=DINO, num_blocks=2,
        num_heads=16, num_tokens=T, token_channels=TOK, zero_init=True, use_fp16=False,
    )
    model = ActionLatentTokenizerV4(encoder=enc, recon_decoder=dec, dino_decoder=dino_dec)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    torch.save({"model_state_dict": model.state_dict()}, path)
    return path


def main():
    # ---- import checks (Phase-2 touched files) ----
    for mod in [
        "gr00t.data.dataset_actlat_fm_v4",
        "gr00t.model.gr00t_n1_actlat_fm",
        "gr00t.experiment.trainer_actlat_fm",
    ]:
        importlib.import_module(mod)
    # finetune script import (module exec without running main)
    spec = importlib.util.spec_from_file_location("ft", "scripts/gr00t_finetune_actlat_fm.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert hasattr(m.ArgsConfig, "__dataclass_fields__")
    assert "actlat_frames" in m.ArgsConfig.__dataclass_fields__
    print("[imports] dataset_actlat_fm_v4 / gr00t_n1_actlat_fm / trainer / finetune OK; --actlat-frames present")

    # ---- wrapper frame path with fake DINO ----
    ckpt = build_and_save()
    wrapper = ActionLatentTokenizerWrapper.from_checkpoint(ckpt, device="cpu")
    wrapper._dino_extractor = _FakeDino()  # inject; bypass real DINO load

    actions = torch.randn(B, T, D)
    frames = (torch.rand(B, 3, 224, 224) * 255).to(torch.uint8)  # uint8 [B,3,H,W]

    g, t, h = wrapper.encode(actions, x0=frames, x1=frames)
    assert t.shape == (B, T, TOK), t.shape
    latent = wrapper.get_latent_target(actions, target_tokens="all", x0=frames, x1=frames)
    assert latent.shape == (B, T, TOK), latent.shape
    dec_actions = wrapper.decode_latent(latent, target_tokens="all")
    assert dec_actions.shape == (B, T, D), dec_actions.shape

    # x0==x1 → dino_diff is zero, but latent must still be finite and well-shaped.
    assert torch.isfinite(latent).all()
    print("[wrapper frame path] get_latent_target(x0,x1 raw frames) → %s; decode → %s OK"
          % (tuple(latent.shape), tuple(dec_actions.shape)))

    # ---- v4 requires frames: missing frames must raise ----
    try:
        wrapper.get_latent_target(actions, target_tokens="all")
        raise AssertionError("expected ValueError when v4 called without frames")
    except ValueError:
        print("[guard] v4 get_latent_target without frames raises ValueError OK")

    print("\nALL PHASE-2 CPU SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
