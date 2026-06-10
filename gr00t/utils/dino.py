"""DINOv3 feature extractor, copied from ``rla-wm/utils/dino.py``.

Structural 1:1 copy of the extractor used by RLA's DINO inverse-dynamics
autoencoder. The visualization helpers and ``__main__`` demo from the original
are intentionally omitted (they pull in matplotlib/sklearn and are not needed for
training). The ``DINOv3FeatureExtractor`` behavior is identical:

  - input ``[B, 3, H, W]`` in ``[0, 1]`` range,
  - internal ImageNet normalization + reflect-pad to a multiple of patch_size,
  - fp16 autocast forward, frozen params, permanent eval mode,
  - returns ``(cls_token[B, D], patch_tokens)`` where patch_tokens is
    ``[B, D, H', W']`` (spatial grid) or ``[B, N, D]``.
"""

from enum import Enum
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel


class DINOv3Model(Enum):
    """Official DINOv3 ViT model identifiers on Hugging Face (LVD-1689M)."""

    SMALL = "facebook/dinov3-vits16-pretrain-lvd1689m"
    BASE = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    LARGE = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    HUGE = "facebook/dinov3-vith16plus-pretrain-lvd1689m"
    GIANT_7B = "facebook/dinov3-vit7b16-pretrain-lvd1689m"


def get_dinov3_model_for_channels(vit_channels: int) -> DINOv3Model:
    """Select the appropriate DINOv3 model based on embedding dimension.

    Typical embedding dimensions:
        SMALL: 384, BASE: 768, LARGE: 1024, HUGE: 1280, GIANT_7B: 1536
    """
    if vit_channels <= 384:
        return DINOv3Model.SMALL
    elif vit_channels <= 768:
        return DINOv3Model.BASE
    elif vit_channels <= 1024:
        return DINOv3Model.LARGE
    elif vit_channels <= 1280:
        return DINOv3Model.HUGE
    else:
        return DINOv3Model.GIANT_7B


class DINOv3FeatureExtractor(nn.Module):
    """Wrapper for extracting features with Meta's DINOv3 ViT models.

    Always runs in eval mode with no gradients and uses float16 internally.
    """

    def __init__(
        self,
        model_name: Union[DINOv3Model, str] = DINOv3Model.SMALL,
        use_compile: bool = True,
        attn_implementation: str = "sdpa",
        final_norm: str = "affine",
    ):
        super().__init__()

        assert final_norm in ("affine", "naive"), (
            f"final_norm must be 'affine' or 'naive'; got {final_norm!r}"
        )
        # "affine" (default): use the model's last_hidden_state, i.e. the final
        # LayerNorm WITH its learned weight/bias (standard DINO behavior).
        # "naive": apply a non-affine LayerNorm ((x-mean)/std, no learned γ/β) to
        # the last encoder hidden state instead — i.e. drop the final LN's affine.
        self.final_norm = final_norm

        self.model_name_str = (
            model_name.value if isinstance(model_name, DINOv3Model) else model_name
        )

        print(f"Loading DINOv3 model: {self.model_name_str}...")
        try:
            self.config = AutoConfig.from_pretrained(self.model_name_str)
            self.model = AutoModel.from_pretrained(
                self.model_name_str,
                config=self.config,
                attn_implementation=attn_implementation,
            )
        except (OSError, KeyError, ValueError) as e:
            # Fallback for testing if the specific DINOv3 repo is private/unavailable.
            print(
                f"Warning: Could not load {self.model_name_str} (error: {e}). "
                "Loading DINOv2 for demo purposes."
            )
            self.model_name_str = "facebook/dinov2-small"
            self.config = AutoConfig.from_pretrained(self.model_name_str)
            self.model = AutoModel.from_pretrained(
                self.model_name_str, config=self.config
            )

        # Set to eval mode permanently
        self.model.eval()

        # Freeze model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        if use_compile:
            print("Compiling DINOv3 model with torch.compile...")
            self.model = torch.compile(self.model)

        # Extract architectural details
        self.patch_size = getattr(self.config, "patch_size", 16)
        self.embed_dim = self.config.hidden_size

        # ImageNet normalization constants
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def train(self, mode: bool = True) -> "DINOv3FeatureExtractor":
        """Override train to always keep model in eval mode."""
        return super().train(False)

    def _preprocess(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        # 1. Ensure float and normalize
        x = x.float()
        x = (x - self.mean) / self.std

        # 2. Pad to multiple of patch_size
        B, C, H, W = x.shape
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        return x, H, W

    @torch.inference_mode()
    def forward(
        self, x: torch.Tensor, return_spatial_grid: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract DINO features from input images.

        Args:
            x: Input tensor of shape [B, C, H, W], values in [0, 1] range.
            return_spatial_grid: If True, reshape patch tokens to [B, D, H', W'].

        Returns:
            cls_token: CLS token features [B, D] (float16).
            patch_tokens: [B, D, H', W'] if return_spatial_grid else [B, N, D].
        """
        x_padded, H_orig, W_orig = self._preprocess(x)

        device_type = "cuda" if x_padded.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, dtype=torch.float16):
            outputs = self.model(x_padded, output_hidden_states=True, return_dict=True)
            if self.final_norm == "naive":
                # Drop the final LayerNorm's learned affine: normalize the last
                # encoder hidden state (pre-final-LN) with a plain non-affine LN.
                base = outputs.hidden_states[-1]
                eps = getattr(self.config, "layer_norm_eps", 1e-6)
                last_hidden_state = F.layer_norm(
                    base.float(), (base.shape[-1],), eps=eps
                ).to(base.dtype)
            else:
                last_hidden_state = outputs.last_hidden_state

        B, N, D = last_hidden_state.shape
        H_padded, W_padded = x_padded.shape[2], x_padded.shape[3]
        hH, hW = H_padded // self.patch_size, W_padded // self.patch_size
        num_patches = hH * hW
        num_extra_tokens = N - num_patches

        cls_token = last_hidden_state[:, 0, :]
        patch_tokens = last_hidden_state[:, num_extra_tokens:, :]

        if return_spatial_grid:
            patch_tokens = patch_tokens.permute(0, 2, 1).reshape(B, D, hH, hW)

        return cls_token, patch_tokens

    @torch.inference_mode()
    def extract_cls_token(self, x: torch.Tensor) -> torch.Tensor:
        """Extract only the CLS token from input images ([B, C, H, W] in [0, 1])."""
        cls_token, _ = self.forward(x, return_spatial_grid=False)
        return cls_token
