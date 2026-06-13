"""VGGT patch-feature extractor (drop-in alternative to ``DINOv3FeatureExtractor``).

Mirrors the API/behavior of :class:`gr00t.utils.dino.DINOv3FeatureExtractor` so the
V4 action tokenizer can swap its frozen visual feature source from DINO to VGGT
without touching the encoder/decoder. Like the DINO extractor, this is:

  - input ``[B, 3, H, W]`` in ``[0, 1]`` range,
  - frozen params, permanent eval mode, no-grad fp16 forward,
  - returns per-frame patch tokens ``[B, Lp, C]`` (each frame run independently as
    a single-view VGGT sequence, S=1 — the direct analog of per-frame DINO feats).

Two token sources (selected at construction):
  - ``"aggregator"``: the VGGT backbone's LAST aggregated layer patch tokens
    (``aggregated_tokens_list[-1]``), ``C = 2*embed_dim = 2048`` — the closest
    analog to DINO's ``last_hidden_state``.
  - ``"dpt_out2"``: the point head's DPT intermediate feature at patch-grid
    resolution (``intermediate_layer_idx[2]`` → layer 17, projected + pos-embedded),
    ``C = out_channels[2] = 1024`` — geometry-biased, same channel count as
    dinov2-large so it is a 1:1 shape replacement.

The VGGT source lives at ``<repo>/vggt`` and is NOT pip-installed, so we inject that
directory onto ``sys.path`` before importing (it is a PEP-420 namespace package).
"""

import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _import_vggt():
    """Import the VGGT class from the in-repo ``vggt/`` source tree.

    ``<repo>/vggt`` is prepended to ``sys.path`` so the inner ``vggt`` namespace
    package (``vggt/vggt/models/vggt.py``) resolves to ``import vggt.models.vggt``.
    """
    repo_root = Path(__file__).resolve().parents[2]  # utils -> gr00t -> repo root
    vggt_root = repo_root / "vggt"
    if str(vggt_root) not in sys.path:
        sys.path.insert(0, str(vggt_root))
    from vggt.models.vggt import VGGT  # noqa: E402

    return VGGT


_VALID_TOKEN_SOURCES = ("aggregator", "dpt_out2")


class VGGTFeatureExtractor(nn.Module):
    """Frozen VGGT visual feature extractor returning patch tokens ``[B, Lp, C]``.

    Always runs in eval mode with no gradients and fp16 autocast internally.

    Args:
        model_name: HF repo id for the VGGT checkpoint (e.g. ``facebook/VGGT-1B``).
        token_source: ``"aggregator"`` (2048-d) or ``"dpt_out2"`` (1024-d).
        image_size: square input size handed to VGGT (must be a multiple of 14).
        final_norm: ``"none"`` (default) returns the raw token features. ``"naive"``
            applies an extra non-affine LayerNorm ((x-mean)/std, no learned γ/β) to
            the final token features — the analog of the DINO extractor's "naive"
            final norm, but added on top rather than replacing an existing affine LN.
    """

    def __init__(
        self,
        model_name: str = "facebook/VGGT-1B",
        token_source: str = "dpt_out2",
        image_size: int = 224,
        use_compile: bool = False,
        final_norm: str = "none",
    ):
        super().__init__()

        assert token_source in _VALID_TOKEN_SOURCES, (
            f"token_source must be one of {_VALID_TOKEN_SOURCES}; got {token_source!r}"
        )
        assert final_norm in ("none", "naive"), (
            f"final_norm must be 'none' or 'naive'; got {final_norm!r}"
        )
        self.final_norm = final_norm
        self.patch_size = 14
        assert image_size % self.patch_size == 0, (
            f"image_size ({image_size}) must be a multiple of patch_size "
            f"({self.patch_size}); VGGT requires divisibility."
        )

        self.model_name_str = model_name
        self.token_source = token_source
        self.image_size = image_size

        VGGT = _import_vggt()
        print(f"Loading VGGT model: {self.model_name_str} (token_source={token_source})...")
        # Load the full pretrained model (PyTorchModelHubMixin applies a strict load,
        # so we keep all heads at load time), then drop the heads we do not need to
        # free memory. The aggregator (24x2 attention blocks) dominates the footprint.
        self.model = VGGT.from_pretrained(self.model_name_str)

        self.model.camera_head = None
        self.model.depth_head = None
        self.model.track_head = None
        if token_source == "aggregator":
            self.model.point_head = None
            # camera_token has shape [1, 2, 1, embed_dim]; aggregated tokens concat
            # frame+global attention outputs, so the token width is 2*embed_dim.
            agg_embed_dim = self.model.aggregator.camera_token.shape[-1]
            self.embed_dim = 2 * agg_embed_dim
        else:  # dpt_out2: keep point_head, read its layer-2 projection width
            point_head = self.model.point_head
            assert point_head is not None, "point_head missing; cannot use dpt_out2 source."
            # projects[2] is the 1x1 conv producing out_channels[2] for layer 17.
            self.embed_dim = point_head.projects[2].out_channels

        # Freeze + permanent eval.
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        if use_compile:
            print("Compiling VGGT model with torch.compile...")
            self.model = torch.compile(self.model)

    def train(self, mode: bool = True) -> "VGGTFeatureExtractor":
        """Override train to always keep the model in eval mode."""
        return super().train(False)

    def _resize(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = F.interpolate(
                x, size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False,
            )
        return x

    @torch.inference_mode()
    def forward(
        self, x: torch.Tensor, return_spatial_grid: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract VGGT patch tokens.

        Args:
            x: input ``[B, 3, H, W]`` in ``[0, 1]``. Resized to ``image_size`` if needed.
            return_spatial_grid: if True, also return the ``[B, C, h, w]`` grid; the
                first element is always the token sequence ``[B, Lp, C]``. Kept for
                API parity with the DINO extractor (callers use the token sequence).

        Returns:
            tokens ``[B, Lp, C]`` and (optionally) the spatial grid ``[B, C, h, w]``.
        """
        x = x.float()
        x = self._resize(x)
        B, _, H, W = x.shape
        images = x.unsqueeze(1)  # [B, S=1, 3, H, W]

        device_type = "cuda" if x.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, dtype=torch.float16):
            tokens_list, patch_start_idx = self.model.aggregator(images)

        if self.token_source == "aggregator":
            # [B, S=1, P, 2C] -> drop frame axis, drop special tokens -> [B, Lp, 2C]
            tok = tokens_list[-1][:, 0, patch_start_idx:, :]
            grid = None
        else:
            tok, grid = self._dpt_out2(tokens_list, patch_start_idx, H, W)

        tok = tok.float()
        if self.final_norm == "naive":
            # Extra non-affine LayerNorm over the channel dim of the final token
            # features (no learned γ/β). Applied to the token sequence the callers
            # consume; the optional spatial grid is normalized to match.
            tok = F.layer_norm(tok, (tok.shape[-1],), eps=1e-6)
            if grid is not None:
                # grid is [B, C, h, w]; normalize over the channel dim (dim=1).
                g = grid.float().permute(0, 2, 3, 1)
                g = F.layer_norm(g, (g.shape[-1],), eps=1e-6)
                grid = g.permute(0, 3, 1, 2)
        if return_spatial_grid:
            return tok, grid
        return tok, None

    def _dpt_out2(self, tokens_list, patch_start_idx, H, W):
        """Reproduce the point head's DPT intermediate feature at layer index 2.

        Mirrors the per-layer ops in ``DPTHead._forward_impl`` (vggt/heads/dpt_head.py)
        for the third intermediate layer (``intermediate_layer_idx[2]`` == 17):
        norm -> 2D reshape -> projects[2] -> (pos embed) -> resize_layers[2] (Identity).
        Returns (tokens ``[B, Lp, 1024]``, grid ``[B, 1024, h, w]``).
        """
        head = self.model.point_head
        dpt_idx = 2
        layer_idx = head.intermediate_layer_idx[dpt_idx]
        ph, pw = H // self.patch_size, W // self.patch_size

        x = tokens_list[layer_idx][:, :, patch_start_idx:]      # [B, 1, Lp, 2C]
        B = x.shape[0]
        x = x.reshape(B, -1, x.shape[-1])                       # [B, Lp, 2C] (S=1)
        x = head.norm(x)
        x = x.permute(0, 2, 1).reshape(B, x.shape[-1], ph, pw)  # [B, 2C, ph, pw]
        x = head.projects[dpt_idx](x)                           # [B, 1024, ph, pw]
        if head.pos_embed:
            x = head._apply_pos_embed(x, W, H)
        x = head.resize_layers[dpt_idx](x)                      # Identity -> [B,1024,ph,pw]

        grid = x
        tok = x.flatten(2).transpose(1, 2)                      # [B, Lp, 1024]
        return tok, grid
