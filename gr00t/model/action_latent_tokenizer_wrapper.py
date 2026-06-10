"""Frozen wrapper around a full ActionLatentTokenizer (encoder + decoder).

Loads a pretrained ActionLatentTokenizer checkpoint, freezes all parameters,
and exposes encode / decode / get_latent_target for use in VLA training and inference.

Designed to be architecture-agnostic: detects the tokenizer type from checkpoint
keys and instantiates the correct model.

Supported types:
- "timewise": TimeWiseEncoder (action_latent_tokenizer.py)
             Detected by: encoder.action_proj.weight exists + NO encoder._is_dimension_wise
             Latent tokens: Ng + T + Nh  (one token per timestep)
- "dimwise":  DimensionWiseEncoder (action_latent_tokenizer_faster.py)
             Detected by: encoder._is_dimension_wise buffer exists
             Latent tokens: Ng + D + Nh  (one token per action dimension)

Adding new tokenizer types: implement _build_XXX_tokenizer and register it in
_TOKENIZER_DETECTORS inside _build_from_state_dict.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class ActionLatentTokenizerWrapper(nn.Module):

    def __init__(self, tokenizer: nn.Module):
        super().__init__()
        self.tokenizer = tokenizer
        # Cache properties from the encoder
        self.num_global_tokens = tokenizer.encoder.num_global_tokens
        self.num_hand_tokens = tokenizer.encoder.num_hand_tokens
        self.action_dim = tokenizer.encoder.action_dim
        self.action_horizon = tokenizer.encoder.action_horizon
        # The transformer's working hidden size (NOT the latent dim downstream
        # consumers see — see ``emb_dim`` below).
        self.internal_emb_dim = tokenizer.encoder.emb_dim
        # ``emb_dim`` is the OUTPUT latent dimension downstream code consumes
        # (action head ``action_dim``, ``decode_latent`` slicing, etc.). For
        # tokenizers without a bottleneck (v1, v2, v3-no-bottleneck, dimwise)
        # this equals the transformer ``emb_dim``. For v3-with-bottleneck it
        # equals the bottleneck ``token_dim``.
        self.emb_dim = getattr(tokenizer.encoder, "token_dim", tokenizer.encoder.emb_dim)

        # Detect tokenizer type and set num_main_tokens accordingly.
        # num_main_tokens: number of "primary" latent tokens (excluding global/hand).
        #   timewise → action_horizon (one token per timestep)
        #   dimwise  → action_dim     (one token per action dimension)
        try:
            from gr00t.model.action_latent_tokenizer_faster import DimensionWiseActionLatentTokenizer
            if isinstance(tokenizer, DimensionWiseActionLatentTokenizer):
                self.tokenizer_type = "dimwise"
                self.num_main_tokens = self.action_dim
            else:
                self.tokenizer_type = "timewise"
                self.num_main_tokens = self.action_horizon
        except ImportError:
            self.tokenizer_type = "timewise"
            self.num_main_tokens = self.action_horizon

        # Freeze all parameters
        for p in self.parameters():
            p.requires_grad = False

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "cpu",
        head_dim_override: Optional[int] = None,
    ):
        """Load full tokenizer from a HuggingFace Trainer checkpoint or raw .pt file.

        Supports:
        - HF Trainer checkpoint dir (reads model.safetensors)
        - Single .pt / .ckpt file with state_dict

        Args:
            head_dim_override: bypass the auto-detect heuristic and force a
                specific attention head_dim. Use only when you know the
                training-time value and the heuristic is wrong (state_dict
                cannot disambiguate head_dim — see error_notes.md 2026-04-29).
                Default heuristic prefers 64 (the training default).
        """
        import os

        # Resolve checkpoint path
        if os.path.isdir(checkpoint_path):
            safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
            pt_path = os.path.join(checkpoint_path, "pytorch_model.bin")
            if os.path.exists(safetensors_path):
                from safetensors.torch import load_file
                state_dict = load_file(safetensors_path, device=device)
            elif os.path.exists(pt_path):
                state_dict = torch.load(pt_path, map_location=device, weights_only=False)
            else:
                raise FileNotFoundError(
                    f"No model.safetensors or pytorch_model.bin found in {checkpoint_path}"
                )
        else:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            elif "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
            else:
                state_dict = ckpt

        tokenizer = cls._build_from_state_dict(state_dict, head_dim_override=head_dim_override)
        # Filter state_dict to only keys present in the model.
        # Training-only modules (mask_token, hand_pred_decoder, etc.) are
        # excluded from the inference model and must be stripped.
        model_keys = set(tokenizer.state_dict().keys())
        filtered_sd = {k: v for k, v in state_dict.items() if k in model_keys}
        extra_keys = set(state_dict.keys()) - model_keys
        if extra_keys:
            print(f"[ActionLatentTokenizerWrapper] Skipping {len(extra_keys)} training-only keys: "
                  f"{sorted(extra_keys)[:5]}{'...' if len(extra_keys) > 5 else ''}")
        tokenizer.load_state_dict(filtered_sd, strict=True)

        wrapper = cls(tokenizer)
        wrapper.to(device)
        wrapper.eval()
        print(f"[ActionLatentTokenizerWrapper] Loaded from {checkpoint_path}")
        print(
            f"  tokenizer_type={wrapper.tokenizer_type}, "
            f"action_dim={wrapper.action_dim}, action_horizon={wrapper.action_horizon}, "
            f"emb_dim={wrapper.emb_dim} (latent), internal_emb_dim={wrapper.internal_emb_dim} (transformer), "
            f"num_global={wrapper.num_global_tokens}, num_hand={wrapper.num_hand_tokens}, "
            f"num_main_tokens={wrapper.num_main_tokens}"
        )
        return wrapper

    @staticmethod
    def _build_from_state_dict(state_dict: dict, head_dim_override: Optional[int] = None):
        """Detect tokenizer architecture from state_dict keys and build it.

        Detection order (add new detectors here for new tokenizer types):
          0. _is_v4 buffer                → timewise v4 (ActionLatentTokenizerV4, RLA-DINO)
          1. _is_v3 buffer                → timewise v3 (ActionLatentTokenizerV3)
          2. _is_v2 buffer                → timewise v2 (ActionLatentTokenizerV2)
          3. encoder._is_dimension_wise   → dimwise  (DimensionWiseEncoder)
          4. encoder.action_proj.weight   → timewise  (TimeWiseEncoder)
        """
        if "_is_v4" in state_dict:
            return ActionLatentTokenizerWrapper._build_timewise_v4_tokenizer(state_dict, head_dim_override)
        if "_is_v3" in state_dict:
            return ActionLatentTokenizerWrapper._build_timewise_v3_tokenizer(state_dict, head_dim_override)
        elif "_is_v2" in state_dict:
            return ActionLatentTokenizerWrapper._build_timewise_v2_tokenizer(state_dict, head_dim_override)
        elif "encoder._is_dimension_wise" in state_dict:
            return ActionLatentTokenizerWrapper._build_dimwise_tokenizer(state_dict, head_dim_override)
        elif "encoder.action_proj.weight" in state_dict:
            return ActionLatentTokenizerWrapper._build_timewise_tokenizer(state_dict, head_dim_override)
        else:
            raise ValueError(
                "Unknown tokenizer architecture. "
                "Expected '_is_v3' (V3), '_is_v2' (V2), 'encoder._is_dimension_wise' "
                "(DimensionWiseEncoder) or 'encoder.action_proj.weight' (TimeWiseEncoder). "
                f"Got keys: {list(state_dict.keys())[:10]}..."
            )

    @staticmethod
    def _resolve_head_dim(emb_dim: int, override: Optional[int]) -> int:
        """Resolve attention head_dim.

        State_dict alone cannot disambiguate head_dim (QKV/proj weights are
        [E, E] regardless of head count, qk_norm has elementwise_affine=False
        so it stores no params). The training default in
        ``train_action_latent_tokenizer_v2.py`` is ``head_dim=64``, so we try
        64 first, then 128, then 32 — picking the largest divisor that the
        training default would have used. Use ``head_dim_override`` to
        bypass when the heuristic is wrong.
        """
        if override is not None:
            return int(override)
        for hd in [64, 128, 32]:
            if emb_dim % hd == 0:
                return hd
        return 64

    @staticmethod
    def _build_timewise_v3_tokenizer(state_dict: dict, head_dim_override: Optional[int] = None):
        """Build ActionLatentTokenizerV3 from state_dict shapes.

        V3 = V2 + optional encoder output LayerNorm + (training-only) latent
        Gaussian noise + optional VTP-style bottleneck. For inference only the
        encoder + recon_decoder are needed; the noise hook is inactive in
        eval() mode regardless.

        Bottleneck detection: ``encoder.output_down_proj.weight`` is present
        only when ``use_bottleneck=True`` (Identity layers contribute no keys).
        Its shape is ``[token_dim, emb_dim]``.
        """
        from gr00t.model.action_latent_tokenizer_v3 import (
            ActionLatentTokenizerV3,
            ReconDecoderV3,
            TimeWiseEncoderV3,
        )

        emb_dim, action_dim = state_dict["encoder.action_proj.weight"].shape
        action_horizon = state_dict["encoder.time_pos_emb.posembs"].shape[2]

        num_global = 0
        if "encoder.global_tokens" in state_dict:
            num_global = state_dict["encoder.global_tokens"].shape[0]

        num_hand = 0
        if "encoder.hand_tokens" in state_dict:
            num_hand = state_dict["encoder.hand_tokens"].shape[0]

        enc_depth = 0
        for k in state_dict:
            if k.startswith("encoder.transformer.blocks."):
                idx = int(k.split(".")[3])
                enc_depth = max(enc_depth, idx + 1)

        head_dim = ActionLatentTokenizerWrapper._resolve_head_dim(emb_dim, head_dim_override)

        # Detect optional encoder output LayerNorm by presence of its weight.
        output_layernorm = "encoder.output_layernorm.weight" in state_dict

        # Detect optional VTP-style bottleneck. When enabled, the encoder ends
        # with Linear(emb_dim, token_dim) named ``output_down_proj``, and the
        # decoder begins with Linear(token_dim, emb_dim) named ``input_up_proj``.
        # Either key alone is sufficient; we check both for sanity.
        use_bottleneck = "encoder.output_down_proj.weight" in state_dict
        if use_bottleneck:
            token_dim = state_dict["encoder.output_down_proj.weight"].shape[0]
            # Sanity-check decoder side: if mismatched, fail loudly so silent
            # state_dict-shape mismatches don't slip through (cf. memory note
            # ``feedback_silent_arch_mismatch.md``).
            if "recon_decoder.input_up_proj.weight" in state_dict:
                dec_in = state_dict["recon_decoder.input_up_proj.weight"].shape[1]
                if dec_in != token_dim:
                    raise ValueError(
                        f"[timewise_v3] bottleneck token_dim mismatch: "
                        f"encoder.output_down_proj outputs {token_dim} but "
                        f"recon_decoder.input_up_proj expects {dec_in}."
                    )
        else:
            token_dim = emb_dim
            if "recon_decoder.input_up_proj.weight" in state_dict:
                raise ValueError(
                    "[timewise_v3] decoder has input_up_proj but encoder has no "
                    "output_down_proj — inconsistent bottleneck state."
                )

        decoder_num_hand = num_hand
        if num_hand > 0 and "recon_decoder.hand_proj.weight" not in state_dict:
            decoder_num_hand = 0

        hand_in_recon = (decoder_num_hand > 0) or (num_hand == 0)

        print(
            f"[timewise_v3] action_dim={action_dim}, action_horizon={action_horizon}, "
            f"emb_dim={emb_dim}, head_dim={head_dim}, depth={enc_depth}, "
            f"num_global={num_global}, num_hand={num_hand}, "
            f"decoder_num_hand={decoder_num_hand}, hand_in_recon={hand_in_recon}, "
            f"output_layernorm={output_layernorm}, "
            f"use_bottleneck={use_bottleneck}, token_dim={token_dim}"
        )

        encoder = TimeWiseEncoderV3(
            action_dim=action_dim,
            action_horizon=action_horizon,
            emb_dim=emb_dim,
            head_dim=head_dim,
            depth=enc_depth,
            pdropout=0.0,
            num_global_tokens=num_global,
            num_hand_tokens=num_hand,
            output_layernorm=output_layernorm,
            use_bottleneck=use_bottleneck,
            token_dim=token_dim,
        )

        # Decoder discovery (same as v2)
        dec_depth = 0
        for k in state_dict:
            if k.startswith("recon_decoder.transformer.blocks."):
                idx = int(k.split(".")[3])
                dec_depth = max(dec_depth, idx + 1)

        has_cross_attn = any(k.startswith("recon_decoder.decoder.") for k in state_dict)
        decoder_mode = "cross_attention" if has_cross_attn else "self_attention"
        if dec_depth == 0 and has_cross_attn:
            for k in state_dict:
                if k.startswith("recon_decoder.decoder.layers."):
                    idx = int(k.split(".")[3])
                    dec_depth = max(dec_depth, idx + 1)

        print(f"[timewise_v3] decoder_mode={decoder_mode}, dec_depth={dec_depth}")

        recon_decoder = ReconDecoderV3(
            action_dim=action_dim,
            action_horizon=action_horizon,
            emb_dim=emb_dim,
            head_dim=head_dim,
            depth=dec_depth,
            pdropout=0.0,
            decoder_mode=decoder_mode,
            num_global_tokens=num_global,
            num_hand_tokens=decoder_num_hand,
            use_bottleneck=use_bottleneck,
            token_dim=token_dim,
        )

        tokenizer = ActionLatentTokenizerV3(
            encoder=encoder,
            recon_decoder=recon_decoder,
            lambda_recon=1.0,
            hand_in_recon=hand_in_recon,
            latent_noise_std=0.0,  # noise is training-only; inference uses 0.
        )

        return tokenizer

    @staticmethod
    def _build_timewise_v4_tokenizer(state_dict: dict, head_dim_override: Optional[int] = None):
        """Build ActionLatentTokenizerV4 (RLA-DINO hybrid) from state_dict shapes.

        Only the encoder (action encoder + RLA fusion encoder) + recon_decoder are
        rebuilt — the ``dino_decoder`` is a training-only module and is left out
        (its keys get filtered by ``from_checkpoint``). ``encode`` still needs the
        fusion encoder (it consumes DINO feats), so the full ``TimeWiseEncoderV4``
        is constructed.

        Shapes:
          encoder.action_encoder.action_proj.weight   → [emb_dim, action_dim]
          encoder.action_encoder.time_pos_emb.posembs → [1, emb_dim, action_horizon]
          encoder.joint.input_layer.weight            → [fusion_width, dino_dim]
          encoder.joint.out_layer.weight              → [token_dim, fusion_width]
        """
        from gr00t.model.action_latent_tokenizer_v4 import (
            ActionLatentTokenizerV4,
            ReconDecoderV4,
            TimeWiseEncoderV4,
        )

        emb_dim, action_dim = state_dict["encoder.action_encoder.action_proj.weight"].shape
        action_horizon = state_dict["encoder.action_encoder.time_pos_emb.posembs"].shape[2]

        fusion_width, dino_dim = state_dict["encoder.joint.input_layer.weight"].shape
        token_dim = state_dict["encoder.joint.out_layer.weight"].shape[0]

        num_global = 0
        if "encoder.action_encoder.global_tokens" in state_dict:
            num_global = state_dict["encoder.action_encoder.global_tokens"].shape[0]
        num_hand = 0
        if "encoder.action_encoder.hand_tokens" in state_dict:
            num_hand = state_dict["encoder.action_encoder.hand_tokens"].shape[0]

        enc_depth = 0
        for k in state_dict:
            if k.startswith("encoder.action_encoder.transformer.blocks."):
                enc_depth = max(enc_depth, int(k.split(".")[4]) + 1)

        fusion_depth = 0
        for k in state_dict:
            if k.startswith("encoder.joint.blocks."):
                fusion_depth = max(fusion_depth, int(k.split(".")[3]) + 1)

        head_dim = ActionLatentTokenizerWrapper._resolve_head_dim(emb_dim, head_dim_override)
        # MultiheadAttention does not store its head count in the state_dict; the
        # training default uses head_dim 64 for the fusion transformer too.
        fusion_heads = max(1, fusion_width // 64)

        # recon decoder discovery (same logic as v2/v3)
        dec_depth = 0
        for k in state_dict:
            if k.startswith("recon_decoder.transformer.blocks."):
                dec_depth = max(dec_depth, int(k.split(".")[3]) + 1)
        has_cross_attn = any(k.startswith("recon_decoder.decoder.") for k in state_dict)
        decoder_mode = "cross_attention" if has_cross_attn else "self_attention"
        if dec_depth == 0 and has_cross_attn:
            for k in state_dict:
                if k.startswith("recon_decoder.decoder.layers."):
                    dec_depth = max(dec_depth, int(k.split(".")[3]) + 1)

        print(
            f"[timewise_v4] action_dim={action_dim}, action_horizon={action_horizon}, "
            f"emb_dim={emb_dim}, token_dim={token_dim}, dino_dim={dino_dim}, "
            f"fusion_width={fusion_width}, fusion_depth={fusion_depth}, "
            f"enc_depth={enc_depth}, dec_depth={dec_depth}, decoder_mode={decoder_mode}, "
            f"num_global={num_global}, num_hand={num_hand}"
        )

        encoder = TimeWiseEncoderV4(
            action_dim=action_dim,
            action_horizon=action_horizon,
            emb_dim=emb_dim,
            head_dim=head_dim,
            encoder_depth=enc_depth,
            pdropout=0.0,
            num_global_tokens=num_global,
            num_hand_tokens=num_hand,
            dino_dim=dino_dim,
            fusion_width=fusion_width,
            fusion_depth=fusion_depth,
            fusion_heads=fusion_heads,
            token_dim=token_dim,
        )

        recon_decoder = ReconDecoderV4(
            action_dim=action_dim,
            action_horizon=action_horizon,
            emb_dim=emb_dim,
            head_dim=head_dim,
            depth=dec_depth,
            pdropout=0.0,
            decoder_mode=decoder_mode,
            num_global_tokens=num_global,
            num_hand_tokens=num_hand,
            token_dim=token_dim,
        )

        # Visual feature source markers (present only for VGGT-trained tokenizers;
        # DINO checkpoints have none → feature_source stays "dino" and no extra
        # buffers are registered, so the strict load below still matches).
        feature_source = "dino"
        vggt_token_source = vggt_image_size = vggt_model = None
        if "_feature_source" in state_dict:
            from gr00t.model.action_latent_tokenizer_v4 import byte_tensor_to_str

            feature_source = byte_tensor_to_str(state_dict["_feature_source"])
            if feature_source == "vggt":
                vggt_token_source = byte_tensor_to_str(state_dict["_vggt_token_source"])
                vggt_image_size = int(state_dict["_vggt_image_size"].item())
                vggt_model = byte_tensor_to_str(state_dict["_vggt_model"])
                print(
                    f"[timewise_v4] feature_source=vggt, token_source={vggt_token_source}, "
                    f"image_size={vggt_image_size}, model={vggt_model}"
                )

        # DINO final-LayerNorm marker (present only when trained with the non-affine
        # "naive" mode; absent → "affine", the standard/default path).
        dino_final_norm = "affine"
        if "_dino_final_norm" in state_dict:
            from gr00t.model.action_latent_tokenizer_v4 import byte_tensor_to_str

            dino_final_norm = byte_tensor_to_str(state_dict["_dino_final_norm"])
            print(f"[timewise_v4] dino_final_norm={dino_final_norm}")

        # dino_decoder omitted (training-only); its checkpoint keys are filtered.
        tokenizer = ActionLatentTokenizerV4(
            encoder=encoder,
            recon_decoder=recon_decoder,
            dino_decoder=None,
            lambda_recon=1.0,
            lambda_dino=0.0,
            feature_source=feature_source,
            vggt_token_source=vggt_token_source,
            vggt_image_size=vggt_image_size,
            vggt_model=vggt_model,
            dino_final_norm=dino_final_norm,
        )
        return tokenizer

    @staticmethod
    def _build_timewise_v2_tokenizer(state_dict: dict, head_dim_override: Optional[int] = None):
        """Build ActionLatentTokenizerV2 from state_dict shapes.

        For inference, only encoder + recon_decoder are needed.
        Training-only modules (hand_pred_decoder, action_text_encoder, etc.) are omitted.
        """
        from gr00t.model.action_latent_tokenizer_v2 import (
            ActionLatentTokenizerV2,
            TimeWiseEncoder,
            ReconDecoder,
        )

        # Infer encoder config
        emb_dim, action_dim = state_dict["encoder.action_proj.weight"].shape  # [E, D]
        action_horizon = state_dict["encoder.time_pos_emb.posembs"].shape[2]  # [1, E, T]

        num_global = 0
        if "encoder.global_tokens" in state_dict:
            num_global = state_dict["encoder.global_tokens"].shape[0]

        num_hand = 0
        if "encoder.hand_tokens" in state_dict:
            num_hand = state_dict["encoder.hand_tokens"].shape[0]

        enc_depth = 0
        for k in state_dict:
            if k.startswith("encoder.transformer.blocks."):
                idx = int(k.split(".")[3])
                enc_depth = max(enc_depth, idx + 1)

        head_dim = ActionLatentTokenizerWrapper._resolve_head_dim(emb_dim, head_dim_override)

        # Decoder num_hand: detect independently (hand_in_recon=False이면 decoder에 hand_proj 없음)
        decoder_num_hand = num_hand
        if num_hand > 0 and "recon_decoder.hand_proj.weight" not in state_dict:
            decoder_num_hand = 0

        hand_in_recon = (decoder_num_hand > 0) or (num_hand == 0)

        print(f"[timewise_v2] action_dim={action_dim}, action_horizon={action_horizon}, "
              f"emb_dim={emb_dim}, head_dim={head_dim}, depth={enc_depth}, "
              f"num_global={num_global}, num_hand={num_hand}, "
              f"decoder_num_hand={decoder_num_hand}, hand_in_recon={hand_in_recon}")

        encoder = TimeWiseEncoder(
            action_dim=action_dim,
            action_horizon=action_horizon,
            emb_dim=emb_dim,
            head_dim=head_dim,
            depth=enc_depth,
            pdropout=0.0,
            num_global_tokens=num_global,
            num_hand_tokens=num_hand,
        )

        # Infer decoder config
        dec_depth = 0
        for k in state_dict:
            if k.startswith("recon_decoder.transformer.blocks."):
                idx = int(k.split(".")[3])
                dec_depth = max(dec_depth, idx + 1)

        has_cross_attn = any(k.startswith("recon_decoder.decoder.") for k in state_dict)
        decoder_mode = "cross_attention" if has_cross_attn else "self_attention"
        if dec_depth == 0 and has_cross_attn:
            # cross_attention uses nn.TransformerDecoder, count its layers
            for k in state_dict:
                if k.startswith("recon_decoder.decoder.layers."):
                    idx = int(k.split(".")[3])
                    dec_depth = max(dec_depth, idx + 1)

        print(f"[timewise_v2] decoder_mode={decoder_mode}, dec_depth={dec_depth}")

        recon_decoder = ReconDecoder(
            action_dim=action_dim,
            action_horizon=action_horizon,
            emb_dim=emb_dim,
            head_dim=head_dim,
            depth=dec_depth,
            pdropout=0.0,
            decoder_mode=decoder_mode,
            num_global_tokens=num_global,
            num_hand_tokens=decoder_num_hand,
        )

        # Build V2 tokenizer with only encoder + recon_decoder (inference mode).
        # Weight loading is handled by from_checkpoint's filtered load_state_dict.
        tokenizer = ActionLatentTokenizerV2(
            encoder=encoder,
            recon_decoder=recon_decoder,
            lambda_recon=1.0,
            hand_in_recon=hand_in_recon,
        )

        return tokenizer

    @staticmethod
    def _build_timewise_tokenizer(state_dict: dict, head_dim_override: Optional[int] = None):
        """Build ActionLatentTokenizer with TimeWiseEncoder from state_dict shapes."""
        from gr00t.model.action_latent_tokenizer import (
            ActionLatentTokenizer,
            TimeWiseEncoder,
            ReconDecoder,
            MaskedReconDecoder,
            TimestepMasking,
        )

        # Infer encoder config
        emb_dim, action_dim = state_dict["encoder.action_proj.weight"].shape  # [E, D]
        action_horizon = state_dict["encoder.time_pos_emb.posembs"].shape[2]  # [1, E, T]

        num_global = 0
        if "encoder.global_tokens" in state_dict:
            num_global = state_dict["encoder.global_tokens"].shape[0]

        num_hand = 0
        if "encoder.hand_tokens" in state_dict:
            num_hand = state_dict["encoder.hand_tokens"].shape[0]

        enc_depth = 0
        for k in state_dict:
            if k.startswith("encoder.transformer.blocks."):
                idx = int(k.split(".")[3])
                enc_depth = max(enc_depth, idx + 1)

        head_dim = ActionLatentTokenizerWrapper._resolve_head_dim(emb_dim, head_dim_override)

        print(f"[timewise] action_dim={action_dim}, action_horizon={action_horizon}, "
              f"emb_dim={emb_dim}, head_dim={head_dim}, depth={enc_depth}, "
              f"num_global={num_global}, num_hand={num_hand}")

        encoder = TimeWiseEncoder(
            action_dim=action_dim,
            action_horizon=action_horizon,
            emb_dim=emb_dim,
            head_dim=head_dim,
            depth=enc_depth,
            pdropout=0.0,
            num_global_tokens=num_global,
            num_hand_tokens=num_hand,
        )

        dec_depth = 0
        for k in state_dict:
            if k.startswith("recon_decoder.transformer.blocks."):
                idx = int(k.split(".")[3])
                dec_depth = max(dec_depth, idx + 1)

        has_cross_attn = any("cross_attn" in k for k in state_dict if k.startswith("recon_decoder."))
        decoder_mode = "cross_attention" if has_cross_attn else "self_attention"
        print(f"[timewise] decoder_mode={decoder_mode}, dec_depth={dec_depth}")

        recon_decoder = ReconDecoder(
            action_dim=action_dim,
            action_horizon=action_horizon,
            emb_dim=emb_dim,
            head_dim=head_dim,
            depth=dec_depth,
            pdropout=0.0,
            decoder_mode=decoder_mode,
            num_hand_tokens=num_hand,
        )

        masked_recon_decoder = None
        masking = None
        has_masked = any(k.startswith("masked_recon_decoder.") for k in state_dict)
        if has_masked:
            masked_depth = 0
            for k in state_dict:
                if k.startswith("masked_recon_decoder.transformer.blocks."):
                    idx = int(k.split(".")[3])
                    masked_depth = max(masked_depth, idx + 1)

            has_masked_cross = any(
                "cross_attn" in k for k in state_dict if k.startswith("masked_recon_decoder.")
            )
            masked_mode = "cross_attention" if has_masked_cross else "self_attention"
            print(f"[timewise] masked_mode={masked_mode}, masked_depth={masked_depth}")

            masked_recon_decoder = MaskedReconDecoder(
                action_dim=action_dim,
                action_horizon=action_horizon,
                emb_dim=emb_dim,
                head_dim=head_dim,
                depth=masked_depth,
                pdropout=0.0,
                decoder_mode=masked_mode,
                num_global_tokens=num_global,
            )
            mask_mode = "random"
            min_mask_ratio = 0.5
            max_mask_ratio = 0.5
            if "masking._mask_mode_bytes" in state_dict:
                mask_mode = bytes(state_dict["masking._mask_mode_bytes"].tolist()).decode()
            if "masking._min_mask_ratio_buf" in state_dict:
                min_mask_ratio = state_dict["masking._min_mask_ratio_buf"].item()
                max_mask_ratio = state_dict["masking._max_mask_ratio_buf"].item()
            print(f"[timewise] masking: mask_mode={mask_mode}, min={min_mask_ratio}, max={max_mask_ratio}")
            masking = TimestepMasking(
                mask_ratio=min_mask_ratio,
                mask_mode=mask_mode,
                min_mask_ratio=min_mask_ratio,
                max_mask_ratio=max_mask_ratio,
            )

        return ActionLatentTokenizer(
            encoder=encoder,
            recon_decoder=recon_decoder,
            masked_recon_decoder=masked_recon_decoder,
            masking=masking,
            lambda_recon=1.0,
            lambda_masked=1.0 if has_masked else 0.0,
        )

    @staticmethod
    def _build_dimwise_tokenizer(state_dict: dict, head_dim_override: Optional[int] = None):
        """Build DimensionWiseActionLatentTokenizer from state_dict shapes."""
        from gr00t.model.action_latent_tokenizer_faster import (
            DimensionWiseActionLatentTokenizer,
            DimensionWiseEncoder,
            DimensionWiseReconDecoder,
            DimensionWiseMaskedReconDecoder,
            DimensionMasking,
        )

        # Infer encoder config
        # action_proj: Linear(T→E), weight shape [E, T]
        emb_dim = state_dict["encoder.action_proj.weight"].shape[0]
        action_horizon = state_dict["encoder.action_proj.weight"].shape[1]
        # dim_pos_emb.posembs: [1, E, D]
        action_dim = state_dict["encoder.dim_pos_emb.posembs"].shape[2]

        num_global = 0
        if "encoder.global_tokens" in state_dict:
            num_global = state_dict["encoder.global_tokens"].shape[0]

        num_hand = 0
        if "encoder.hand_tokens" in state_dict:
            num_hand = state_dict["encoder.hand_tokens"].shape[0]

        enc_depth = 0
        for k in state_dict:
            if k.startswith("encoder.transformer.blocks."):
                idx = int(k.split(".")[3])
                enc_depth = max(enc_depth, idx + 1)

        head_dim = ActionLatentTokenizerWrapper._resolve_head_dim(emb_dim, head_dim_override)

        print(f"[dimwise] action_dim={action_dim}, action_horizon={action_horizon}, "
              f"emb_dim={emb_dim}, head_dim={head_dim}, depth={enc_depth}, "
              f"num_global={num_global}, num_hand={num_hand}")

        encoder = DimensionWiseEncoder(
            action_dim=action_dim,
            action_horizon=action_horizon,
            emb_dim=emb_dim,
            head_dim=head_dim,
            depth=enc_depth,
            pdropout=0.0,
            num_global_tokens=num_global,
            num_hand_tokens=num_hand,
        )

        dec_depth = 0
        for k in state_dict:
            if k.startswith("recon_decoder.transformer.blocks."):
                idx = int(k.split(".")[3])
                dec_depth = max(dec_depth, idx + 1)

        has_cross_attn = any("cross_attn" in k for k in state_dict if k.startswith("recon_decoder."))
        decoder_mode = "cross_attention" if has_cross_attn else "self_attention"
        print(f"[dimwise] decoder_mode={decoder_mode}, dec_depth={dec_depth}")

        recon_decoder = DimensionWiseReconDecoder(
            action_dim=action_dim,
            action_horizon=action_horizon,
            emb_dim=emb_dim,
            head_dim=head_dim,
            depth=dec_depth,
            pdropout=0.0,
            decoder_mode=decoder_mode,
            num_hand_tokens=num_hand,
        )

        masked_recon_decoder = None
        masking = None
        has_masked = any(k.startswith("masked_recon_decoder.") for k in state_dict)
        if has_masked:
            masked_depth = 0
            for k in state_dict:
                if k.startswith("masked_recon_decoder.transformer.blocks."):
                    idx = int(k.split(".")[3])
                    masked_depth = max(masked_depth, idx + 1)

            has_masked_cross = any(
                "cross_attn" in k for k in state_dict if k.startswith("masked_recon_decoder.")
            )
            masked_mode = "cross_attention" if has_masked_cross else "self_attention"
            print(f"[dimwise] masked_mode={masked_mode}, masked_depth={masked_depth}")

            masked_recon_decoder = DimensionWiseMaskedReconDecoder(
                action_dim=action_dim,
                action_horizon=action_horizon,
                emb_dim=emb_dim,
                head_dim=head_dim,
                depth=masked_depth,
                pdropout=0.0,
                decoder_mode=masked_mode,
                num_global_tokens=num_global,
            )
            mask_mode = "random"
            min_mask_ratio = 0.5
            max_mask_ratio = 0.5
            if "masking._mask_mode_bytes" in state_dict:
                mask_mode = bytes(state_dict["masking._mask_mode_bytes"].tolist()).decode()
            if "masking._min_mask_ratio_buf" in state_dict:
                min_mask_ratio = state_dict["masking._min_mask_ratio_buf"].item()
                max_mask_ratio = state_dict["masking._max_mask_ratio_buf"].item()
            print(f"[dimwise] masking: mask_mode={mask_mode}, min={min_mask_ratio}, max={max_mask_ratio}")
            masking = DimensionMasking(
                mask_ratio=min_mask_ratio,
                mask_mode=mask_mode,
                min_mask_ratio=min_mask_ratio,
                max_mask_ratio=max_mask_ratio,
            )

        return DimensionWiseActionLatentTokenizer(
            encoder=encoder,
            recon_decoder=recon_decoder,
            masked_recon_decoder=masked_recon_decoder,
            masking=masking,
            lambda_recon=1.0,
            lambda_masked=1.0 if has_masked else 0.0,
        )

    def get_num_tokens(self, target_tokens: str = "all") -> int:
        """Return number of output latent tokens for a given target mode.

        target_tokens:
          "time"        → num_main_tokens (T for timewise, D for dimwise)
          "global_time" → num_global + num_main_tokens
          "time_hand"   → num_main_tokens + num_hand_tokens
          "all"         → num_global + num_main_tokens + num_hand_tokens
        """
        n = self.num_main_tokens
        if target_tokens == "time":
            return n
        elif target_tokens == "global_time":
            return self.num_global_tokens + n
        elif target_tokens == "time_hand":
            return n + self.num_hand_tokens
        elif target_tokens == "all":
            return self.num_global_tokens + n + self.num_hand_tokens
        else:
            raise ValueError(f"Unknown target_tokens mode: {target_tokens}")

    def _is_v4(self) -> bool:
        return hasattr(self.tokenizer, "_is_v4")

    def _ensure_dino_extractor(self, device):
        """Lazily build a frozen DINO extractor for the V4 encode path.

        Stored as a plain attribute (NOT a registered submodule) so it never
        enters this wrapper's state_dict. The DINO variant is derived from the
        encoder's input channel count (``dino_dim``).
        """
        if getattr(self, "_dino_extractor", None) is None:
            from gr00t.utils.dino import DINOv3FeatureExtractor

            dino_dim = self.tokenizer.encoder.dino_dim
            # Stage-1 trains the V4 tokenizer against dinov2 features (transformers
            # 4.51.3 predates dinov3), so Stage-2 MUST use the matching dinov2
            # model — NOT get_dinov3_model_for_channels (which returns a dinov3
            # repo that fails to load here and silently falls back to dinov2-small
            # → 384-dim → shape mismatch). Pick the dinov2 variant by channel.
            dinov2_by_channels = {
                384: "facebook/dinov2-small",
                768: "facebook/dinov2-base",
                1024: "facebook/dinov2-large",
                1536: "facebook/dinov2-giant",
            }
            model = dinov2_by_channels.get(dino_dim)
            assert model is not None, (
                f"No dinov2 model for dino_dim={dino_dim}; "
                f"known: {sorted(dinov2_by_channels)}"
            )
            extractor = DINOv3FeatureExtractor(
                model_name=model,
                use_compile=False,
                final_norm=getattr(self.tokenizer, "dino_final_norm", "affine"),
            )
            assert extractor.embed_dim == dino_dim, (
                f"DINO embed_dim={extractor.embed_dim} != encoder dino_dim={dino_dim} "
                f"(model={model}). A silent fallback likely occurred."
            )
            extractor.eval()
            for p in extractor.parameters():
                p.requires_grad = False
            extractor.to(device)
            self._dino_extractor = extractor
        return self._dino_extractor

    def _ensure_vggt_extractor(self, device):
        """Lazily build a frozen VGGT extractor for the V4 encode path (vggt source).

        Mirrors ``_ensure_dino_extractor``: stored as a plain attribute (not a
        submodule) so it never enters this wrapper's state_dict. Config (token
        source / image size / model id) comes from the markers recorded at training
        time and decoded into the tokenizer at build.
        """
        if getattr(self, "_vggt_extractor", None) is None:
            from gr00t.model.action_latent_tokenizer_v4 import byte_tensor_to_str
            from gr00t.utils.vggt_feature import VGGTFeatureExtractor

            tok = self.tokenizer
            token_source = byte_tensor_to_str(tok._vggt_token_source)
            image_size = int(tok._vggt_image_size.item())
            model_name = byte_tensor_to_str(tok._vggt_model)
            extractor = VGGTFeatureExtractor(
                model_name=model_name,
                token_source=token_source,
                image_size=image_size,
                use_compile=False,
            )
            assert extractor.embed_dim == tok.encoder.dino_dim, (
                f"VGGT embed_dim={extractor.embed_dim} != encoder dino_dim="
                f"{tok.encoder.dino_dim} (token_source={token_source})."
            )
            extractor.eval()
            for p in extractor.parameters():
                p.requires_grad = False
            extractor.to(device)
            self._vggt_extractor = extractor
        return self._vggt_extractor

    @torch.no_grad()
    def _resolve_dino_feats(self, x0, x1, x0_feat, x1_feat, device):
        """Return (x0_feat, x1_feat) [B, Lp, C] from precomputed feats or raw frames."""
        if x0_feat is not None and x1_feat is not None:
            return x0_feat.to(device), x1_feat.to(device)
        if x0 is None or x1 is None:
            raise ValueError(
                "V4 encode requires visual features: pass x0_feat/x1_feat or raw "
                "frames x0/x1 ([B,3,H,W])."
            )

        if getattr(self.tokenizer, "feature_source", "dino") == "vggt":
            extractor = self._ensure_vggt_extractor(device)

            def to_feat(frames):
                frames = frames.to(device).float()
                if frames.max() > 1.5:
                    frames = frames / 255.0
                tok, _ = extractor(frames)  # [B, Lp, C]
                return tok.float()

            return to_feat(x0), to_feat(x1)

        extractor = self._ensure_dino_extractor(device)

        def to_feat(frames):
            frames = frames.to(device).float()
            if frames.max() > 1.5:
                frames = frames / 255.0
            _, grid = extractor(frames, return_spatial_grid=True)  # [B, C, h, w]
            return grid.flatten(2).transpose(1, 2).float()         # [B, h*w, C]

        return to_feat(x0), to_feat(x1)

    @torch.no_grad()
    def encode(
        self,
        actions: torch.Tensor,
        x0=None,
        x1=None,
        x0_feat=None,
        x1_feat=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode actions to (global_tok, main_tok, hand_tok).

        Args:
            actions: [B, T, D] already normalized to [-1, 1]
            x0/x1: optional raw frames [B,3,H,W] (current/future), V4 only.
            x0_feat/x1_feat: optional precomputed DINO feats [B,Lp,C], V4 only.
        Returns:
            timewise: (global[B,Ng,E], time[B,T,E],  hand[B,Nh,E])
            dimwise:  (global[B,Ng,E], dim[B,D,E],   hand[B,Nh,E])
        """
        dtype = self.tokenizer.encoder.action_proj.weight.dtype
        actions = actions.to(dtype=dtype)
        if self._is_v4():
            f0, f1 = self._resolve_dino_feats(x0, x1, x0_feat, x1_feat, actions.device)
            return self.tokenizer.encode(actions, f0, f1)
        return self.tokenizer.encode(actions)

    @torch.no_grad()
    def get_latent_target(
        self,
        actions: torch.Tensor,
        target_tokens: str = "all",
        x0=None,
        x1=None,
        x0_feat=None,
        x1_feat=None,
    ) -> torch.Tensor:
        """Encode actions and return concatenated latent target.

        Args:
            actions: [B, T, D] already normalized to [-1, 1]
            target_tokens: "time", "global_time", "time_hand", "all"
            x0/x1/x0_feat/x1_feat: V4-only DINO inputs (ignored for v2/v3).
        Returns:
            [B, N, E] concatenated latent tokens
        """
        global_tok, main_tok, hand_tok = self.encode(
            actions, x0=x0, x1=x1, x0_feat=x0_feat, x1_feat=x1_feat
        )

        parts = []
        if target_tokens in ("global_time", "all") and self.num_global_tokens > 0:
            parts.append(global_tok)
        parts.append(main_tok)
        if target_tokens in ("time_hand", "all") and self.num_hand_tokens > 0:
            parts.append(hand_tok)

        return torch.cat(parts, dim=1).detach()

    @torch.no_grad()
    def decode_latent(
        self, latent: torch.Tensor, target_tokens: str = "all"
    ) -> torch.Tensor:
        """Decode concatenated latent tokens back to action space.

        Args:
            latent: [B, N, self.emb_dim] concatenated latent tokens (same order
                as ``get_latent_target``). For v3 with bottleneck, the per-token
                feature dim is the bottleneck ``token_dim``, NOT the
                transformer ``internal_emb_dim`` — the decoder up-projects
                internally.
            target_tokens: must match what was used for ``get_latent_target``.
        Returns:
            [B, T, D] reconstructed actions
        """
        Ng = self.num_global_tokens
        main_n = self.num_main_tokens
        Nh = self.num_hand_tokens

        idx = 0
        if target_tokens in ("global_time", "all") and Ng > 0:
            global_tok = latent[:, idx:idx + Ng]
            idx += Ng
        else:
            global_tok = torch.zeros(
                latent.shape[0], 0, self.emb_dim, device=latent.device, dtype=latent.dtype
            )

        main_tok = latent[:, idx:idx + main_n]
        idx += main_n

        if target_tokens in ("time_hand", "all") and Nh > 0:
            hand_tok = latent[:, idx:idx + Nh]
        else:
            hand_tok = torch.zeros(
                latent.shape[0], 0, self.emb_dim, device=latent.device, dtype=latent.dtype
            )

        return self.tokenizer.decode(global_tok, main_tok, hand_tok)
