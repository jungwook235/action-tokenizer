import torch
import torch.nn as nn
from typing import Optional


class ActionLatentEncoderWrapper(nn.Module):
    """Frozen wrapper around the OAT ActionLatentModel encoder.

    Loads a pretrained ActionLatentModel checkpoint, extracts the encoder,
    freezes all parameters, and exposes encode() with token selection.

    Actions from GR00T's data pipeline are already min-max normalized to [-1, 1],
    which matches the ActionLatentModel's normalizer output, so we skip normalization
    and feed actions directly to the encoder.
    """

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        # Cache counts from encoder
        self.num_global_tokens = encoder.num_global_tokens
        self.num_hand_tokens = encoder.num_hand_tokens
        self.action_dim = encoder.action_dim
        self.emb_dim = encoder.emb_dim
        # Freeze
        for p in self.parameters():
            p.requires_grad = False

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: str = "cpu"):
        """Load encoder from a full ActionLatentModel checkpoint.

        Args:
            checkpoint_path: path to .pt/.ckpt file containing ActionLatentModel state_dict
                             (expects keys like 'encoder.xxx', 'normalizer.xxx', etc.)
            device: device to load onto
        """
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Support both raw state_dict and wrapped checkpoint formats
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt

        # Extract encoder config from checkpoint metadata if available
        if "encoder_config" in ckpt:
            enc_cfg = ckpt["encoder_config"]
        else:
            # Infer from encoder weights
            enc_cfg = cls._infer_encoder_config(state_dict)

        from oat.action_latent.v1.encoder.dimwise_encoder import DimensionWiseEncoder

        encoder = DimensionWiseEncoder(**enc_cfg)

        # Load encoder weights (filter prefix)
        encoder_state = {}
        for k, v in state_dict.items():
            if k.startswith("encoder."):
                encoder_state[k[len("encoder."):]] = v
        encoder.load_state_dict(encoder_state, strict=True)

        wrapper = cls(encoder)
        wrapper.eval()
        return wrapper

    @staticmethod
    def _infer_encoder_config(state_dict: dict) -> dict:
        """Infer DimensionWiseEncoder constructor args from state_dict shapes."""
        # temporal_proj.weight: [emb_dim, action_horizon]
        tp_weight = state_dict["encoder.temporal_proj.weight"]
        emb_dim = tp_weight.shape[0]
        action_horizon = tp_weight.shape[1]

        # dim_pos_emb has emb with shape related to action_dim
        # global_tokens: [num_global, emb_dim] if exists
        num_global = 0
        if "encoder.global_tokens" in state_dict:
            num_global = state_dict["encoder.global_tokens"].shape[0]

        num_hand = 0
        if "encoder.hand_tokens" in state_dict:
            num_hand = state_dict["encoder.hand_tokens"].shape[0]

        # Count transformer depth from block keys
        block_indices = set()
        for k in state_dict:
            if k.startswith("encoder.transformer.blocks."):
                idx = int(k.split(".")[3])
                block_indices.add(idx)
        depth = len(block_indices) if block_indices else 4

        # Infer head_dim from attention weights
        # attention qkv weight shape: [3 * num_heads * head_dim, emb_dim]
        head_dim = 64  # default
        for k in state_dict:
            if "attn.qkv.weight" in k and k.startswith("encoder."):
                qkv_out = state_dict[k].shape[0]
                # qkv_out = 3 * num_heads * head_dim, total = 3 * emb_dim typically
                num_heads = qkv_out // (3 * head_dim)
                if num_heads * 3 * head_dim != qkv_out:
                    # Try common head_dims
                    for hd in [32, 64, 128]:
                        if qkv_out % (3 * hd) == 0:
                            head_dim = hd
                            break
                break

        # Infer action_dim from positional embedding
        action_dim = 7  # fallback
        for k in state_dict:
            if "dim_pos_emb" in k and "emb" in k and k.startswith("encoder."):
                # PositionalEmbeddingAdder stores emb of shape [1, max_size, emb_dim] or similar
                shape = state_dict[k].shape
                if len(shape) >= 2:
                    action_dim = shape[-2] if shape[-2] != emb_dim else shape[-1]
                break

        return {
            "action_dim": action_dim,
            "action_horizon": action_horizon,
            "emb_dim": emb_dim,
            "head_dim": head_dim,
            "depth": depth,
            "pdropout": 0.0,
            "num_global_tokens": num_global,
            "num_hand_tokens": num_hand,
        }

    def get_num_tokens(self, target_tokens: str) -> int:
        """Return number of output tokens for a given target mode."""
        if target_tokens == "dim":
            return self.action_dim
        elif target_tokens == "global_dim":
            return self.num_global_tokens + self.action_dim
        elif target_tokens == "dim_hand":
            return self.action_dim + self.num_hand_tokens
        elif target_tokens == "all":
            return self.num_global_tokens + self.action_dim + self.num_hand_tokens
        else:
            raise ValueError(f"Unknown target_tokens mode: {target_tokens}")

    @torch.no_grad()
    def encode(self, actions: torch.Tensor, target_tokens: str = "dim") -> torch.Tensor:
        """Encode actions and return selected latent tokens.

        Args:
            actions: [B, T, D] already normalized to [-1, 1]
            target_tokens: which tokens to return

        Returns:
            [B, N, E] selected latent tokens (detached, no grad)
        """
        global_tok, dim_tok, hand_tok = self.encoder(actions)

        if target_tokens == "dim":
            return dim_tok.detach()
        elif target_tokens == "global_dim":
            return torch.cat([global_tok, dim_tok], dim=1).detach()
        elif target_tokens == "dim_hand":
            return torch.cat([dim_tok, hand_tok], dim=1).detach()
        elif target_tokens == "all":
            return torch.cat([global_tok, dim_tok, hand_tok], dim=1).detach()
        else:
            raise ValueError(f"Unknown target_tokens mode: {target_tokens}")
