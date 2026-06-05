"""Action Latent Tokenizer V3.

V3 = V2 architecture + three opt-in features:

1. **Optional encoder-output LayerNorm.** ``TimeWiseEncoderV3`` applies
   ``nn.LayerNorm(emb_dim)`` to the encoder transformer output BEFORE the
   tokens are split into (global, time, hand) and fed to any decoder. When
   ``output_layernorm=False`` the layer is :class:`nn.Identity` and the
   state_dict is byte-for-byte compatible with V2.

2. **Optional Gaussian noise on encoded latents.** ``ActionLatentTokenizerV3``
   adds ``N(0, σ²·I)`` to ``(global_tok, time_tok, hand_tok)`` (and to the
   masked-path equivalent ``(m_global, m_time, m_hand)``) before any decoder
   call. Active only during training and only when ``latent_noise_std > 0``.

3. **Optional VTP-style bottleneck.** When ``use_bottleneck=True``, the encoder
   appends a single ``Linear(emb_dim, token_dim)`` (``output_down_proj``) so
   the output latent tokens have dimension ``token_dim`` instead of
   ``emb_dim``. The ``ReconDecoderV3`` adds a matching
   ``Linear(token_dim, emb_dim)`` up-projection before the v2 decoder
   pipeline, and ``HandStatePredDecoderV3`` does the same for its KV stream.
   Default ``token_dim=64``. The downstream wrapper detects the bottleneck
   from state_dict and exposes ``wrapper.emb_dim = token_dim`` so VLA
   training reads the correct latent dimension.

   Note: the bottleneck does NOT include its own LayerNorm. If you want LN
   before the projection (VTP's ``ln_final → projection`` pattern), enable
   ``output_layernorm=True`` — that single LN serves both as the V3
   "encoder-output LN" and as the bottleneck pre-norm. Both options default
   to ``False`` so they are byte-compatible with V2.

Everything else (recon decoder, hand state prediction, masked recon, global
contrastive/regression, frequency loss, mask schedule) is inherited from V2's
:class:`ActionLatentTokenizerV2` so the loss bookkeeping is identical.

The model registers BOTH ``_is_v3`` and ``_is_v2`` buffers. The wrapper checks
``_is_v3`` first; the ``_is_v2`` registration is a fallback so older readers
that only know about V2 can still detect the model family.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from gr00t.model.action_latent_tokenizer_v2 import (
    ActionLatentTokenizerV2,
    ActionTextEncoder,
    GlobalTokenLossModule,
    HandStatePredDecoder,
    PositionalEmbeddingAdder,
    ReconDecoder,
    TimeWiseEncoder,
    Transformer,
    frequency_loss,
)


# =====================================================================
# Encoder with optional output LayerNorm
# =====================================================================


class TimeWiseEncoderV3(nn.Module):
    """V3 encoder = V2 encoder + optional output ``LayerNorm`` + optional
    VTP-style bottleneck at the output.

    Pipeline: ``transformer → output_layernorm → output_down_proj
    → split(global, time, hand)``.

    - ``output_layernorm`` (V3 feature 1): LayerNorm applied to the transformer
      output. ``Identity`` when ``output_layernorm=False`` (default). When
      both ``output_layernorm=True`` and ``use_bottleneck=True`` are enabled,
      this single LN serves as the bottleneck pre-norm — there is no separate
      ``bottleneck_norm`` layer (VTP's ``ln_final → projection``).
    - ``output_down_proj`` (V3 feature 3): a single ``Linear(emb_dim,
      token_dim)`` that reduces the latent dimension. ``Identity`` when
      ``use_bottleneck=False`` (default), in which case ``self.token_dim ==
      emb_dim`` and the state_dict is identical to V3 without the bottleneck.

    The attribute :attr:`token_dim` always reflects the true output dimension
    of the encoder. Downstream code (the wrapper, action heads in VLA training)
    must use ``token_dim`` (or equivalently ``wrapper.emb_dim``) — NOT the
    transformer's working ``emb_dim`` — to size projections that consume the
    output tokens.
    """

    def __init__(
        self,
        action_dim: int,
        action_horizon: int,
        emb_dim: int,
        head_dim: int,
        depth: int,
        pdropout: float,
        num_global_tokens: int = 0,
        num_hand_tokens: int = 0,
        output_layernorm: bool = False,
        use_bottleneck: bool = False,
        token_dim: int = 64,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.emb_dim = emb_dim
        self.num_global_tokens = num_global_tokens
        self.num_hand_tokens = num_hand_tokens

        self.action_proj = nn.Linear(action_dim, emb_dim)
        self.time_pos_emb = PositionalEmbeddingAdder(emb_dim, max_sizes=[action_horizon])

        self.global_tokens = (
            nn.Parameter(torch.randn(num_global_tokens, emb_dim))
            if num_global_tokens > 0
            else None
        )
        self.hand_tokens = (
            nn.Parameter(torch.randn(num_hand_tokens, emb_dim))
            if num_hand_tokens > 0
            else None
        )

        self.transformer = Transformer(
            dim=emb_dim, depth=depth, head_dim=head_dim, drop=pdropout
        )

        # V3 feature 1: optional LayerNorm at transformer output. Default off.
        self.has_output_layernorm = bool(output_layernorm)
        self.output_layernorm = (
            nn.LayerNorm(emb_dim) if output_layernorm else nn.Identity()
        )

        # V3 feature 3: optional bottleneck (single Linear, no own LayerNorm).
        # If LN before the projection is desired, enable ``output_layernorm``
        # — that single LN doubles as the bottleneck pre-norm. Default off.
        self.use_bottleneck = bool(use_bottleneck)
        if self.use_bottleneck:
            self.token_dim = int(token_dim)
            self.output_down_proj = nn.Linear(emb_dim, self.token_dim)
        else:
            self.token_dim = emb_dim
            self.output_down_proj = nn.Identity()

    def encode_postproc(self, x: torch.Tensor) -> torch.Tensor:
        """Apply output LayerNorm + (optional) bottleneck projection.

        Used by both :meth:`forward` and the masked-path forward in
        :class:`ActionLatentTokenizerV3` so the two paths stay in sync.
        """
        x = self.output_layernorm(x)
        x = self.output_down_proj(x)
        return x

    def forward(self, x: torch.Tensor):
        B, T, D = x.shape
        Ng = self.num_global_tokens
        Nh = self.num_hand_tokens

        x = self.action_proj(x)
        x = self.time_pos_emb(x)

        parts = []
        if Ng > 0:
            parts.append(self.global_tokens.unsqueeze(0).expand(B, -1, -1))
        parts.append(x)
        if Nh > 0:
            parts.append(self.hand_tokens.unsqueeze(0).expand(B, -1, -1))
        x = torch.cat(parts, dim=1)

        x = self.transformer(x)
        x = self.encode_postproc(x)  # output_layernorm + (optional) bottleneck

        global_out = x[:, :Ng]
        time_out = x[:, Ng:Ng + T]
        hand_out = x[:, Ng + T:]

        return global_out, time_out, hand_out


# =====================================================================
# Decoders with optional bottleneck up-projection
# =====================================================================


class ReconDecoderV3(ReconDecoder):
    """V2's :class:`ReconDecoder` + optional input up-projection.

    When ``use_bottleneck=True``, expects input tokens (time/global/hand) of
    dimension ``token_dim`` and applies ``Linear(token_dim, emb_dim)`` to each
    stream BEFORE the v2 decoder pipeline (which then runs at ``emb_dim``).
    The single ``input_up_proj`` is shared across the three streams — the
    bottleneck is symmetric on all latent tokens, mirroring the encoder where
    a single ``output_down_proj`` is applied to the concatenated sequence.

    When ``use_bottleneck=False``, ``input_up_proj`` is :class:`nn.Identity`
    and the state_dict + behavior are identical to v2's :class:`ReconDecoder`.
    """

    def __init__(
        self,
        action_dim: int,
        action_horizon: int,
        emb_dim: int,
        head_dim: int,
        depth: int,
        pdropout: float,
        decoder_mode: str = "self_attention",
        num_global_tokens: int = 0,
        num_hand_tokens: int = 0,
        use_bottleneck: bool = False,
        token_dim: int = 64,
    ):
        super().__init__(
            action_dim=action_dim,
            action_horizon=action_horizon,
            emb_dim=emb_dim,
            head_dim=head_dim,
            depth=depth,
            pdropout=pdropout,
            decoder_mode=decoder_mode,
            num_global_tokens=num_global_tokens,
            num_hand_tokens=num_hand_tokens,
        )
        self.use_bottleneck = bool(use_bottleneck)
        if self.use_bottleneck:
            self.token_dim = int(token_dim)
            self.input_up_proj = nn.Linear(self.token_dim, emb_dim)
        else:
            self.token_dim = emb_dim
            self.input_up_proj = nn.Identity()

    def forward(
        self,
        time_tokens: torch.Tensor,
        global_tokens: Optional[torch.Tensor] = None,
        hand_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_bottleneck:
            time_tokens = self.input_up_proj(time_tokens)
            if global_tokens is not None and global_tokens.numel() > 0:
                global_tokens = self.input_up_proj(global_tokens)
            if hand_tokens is not None and hand_tokens.numel() > 0:
                hand_tokens = self.input_up_proj(hand_tokens)
        return super().forward(time_tokens, global_tokens=global_tokens, hand_tokens=hand_tokens)


class HandStatePredDecoderV3(HandStatePredDecoder):
    """V2's :class:`HandStatePredDecoder` + optional KV up-projection.

    The encoder's output (hand or time tokens) feeds into the hand-state
    predictor as the KV stream. When the encoder bottleneck is enabled, those
    tokens have dimension ``token_dim`` instead of ``emb_dim``, so we
    up-project before the v2 pipeline.

    When ``use_bottleneck=False``, ``kv_up_proj`` is :class:`nn.Identity` and
    the state_dict/behavior matches v2's :class:`HandStatePredDecoder`.
    """

    def __init__(
        self,
        hand_state_dim: int,
        emb_dim: int,
        head_dim: int,
        depth: int,
        pdropout: float,
        num_future_steps: int,
        num_kv_tokens: int,
        use_bottleneck: bool = False,
        token_dim: int = 64,
    ):
        super().__init__(
            hand_state_dim=hand_state_dim,
            emb_dim=emb_dim,
            head_dim=head_dim,
            depth=depth,
            pdropout=pdropout,
            num_future_steps=num_future_steps,
            num_kv_tokens=num_kv_tokens,
        )
        self.use_bottleneck = bool(use_bottleneck)
        if self.use_bottleneck:
            self.token_dim = int(token_dim)
            self.kv_up_proj = nn.Linear(self.token_dim, emb_dim)
        else:
            self.token_dim = emb_dim
            self.kv_up_proj = nn.Identity()

    def forward(self, hand_state: torch.Tensor, kv_tokens: torch.Tensor) -> torch.Tensor:
        if self.use_bottleneck:
            kv_tokens = self.kv_up_proj(kv_tokens)
        return super().forward(hand_state, kv_tokens)


# =====================================================================
# Tokenizer V3 = V2 + latent Gaussian noise + V3 marker
# =====================================================================


class ActionLatentTokenizerV3(ActionLatentTokenizerV2):
    """V3 tokenizer.

    Inherits all losses (recon, hand_pred, mask_recon, mask_hand_pred, global,
    freq) from V2. Two additions:

    * encoder may be a :class:`TimeWiseEncoderV3` with output LayerNorm,
    * before any decoder call (recon path AND masked-path recon), Gaussian
      noise ``N(0, σ²·I)`` is added to the latents during training when
      ``latent_noise_std > 0``.
    """

    def __init__(
        self,
        encoder,
        recon_decoder: ReconDecoder,
        hand_pred_decoder: Optional[HandStatePredDecoder] = None,
        action_text_encoder: Optional[ActionTextEncoder] = None,
        global_loss_module: Optional[GlobalTokenLossModule] = None,
        # Loss weights
        lambda_recon: float = 1.0,
        lambda_hand_pred: float = 0.0,
        lambda_mask_recon: float = 0.0,
        lambda_mask_hand_pred: float = 0.0,
        lambda_global: float = 0.0,
        freq_loss_weight: float = 0.0,
        # Mask schedule
        mask_ratio: float = 0.5,
        mask_ratio_min: Optional[float] = None,
        mask_ratio_max: Optional[float] = None,
        mask_mode: str = "random",
        mask_batch_ratio: float = 0.5,
        # Recon config
        recon_loss_type: str = "mse",
        # Hand token config
        hand_in_recon: bool = True,
        state_pred_kv_source: str = "hand",
        # V3 additions
        latent_noise_std: float = 0.0,
    ):
        super().__init__(
            encoder=encoder,
            recon_decoder=recon_decoder,
            hand_pred_decoder=hand_pred_decoder,
            action_text_encoder=action_text_encoder,
            global_loss_module=global_loss_module,
            lambda_recon=lambda_recon,
            lambda_hand_pred=lambda_hand_pred,
            lambda_mask_recon=lambda_mask_recon,
            lambda_mask_hand_pred=lambda_mask_hand_pred,
            lambda_global=lambda_global,
            freq_loss_weight=freq_loss_weight,
            mask_ratio=mask_ratio,
            mask_ratio_min=mask_ratio_min,
            mask_ratio_max=mask_ratio_max,
            mask_mode=mask_mode,
            mask_batch_ratio=mask_batch_ratio,
            recon_loss_type=recon_loss_type,
            hand_in_recon=hand_in_recon,
            state_pred_kv_source=state_pred_kv_source,
        )
        self.latent_noise_std = float(latent_noise_std)
        # V3 detection marker (in addition to inherited _is_v2).
        self.register_buffer("_is_v3", torch.tensor(True))

    def _maybe_noise(self, t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if t is None or t.numel() == 0:
            return t
        if not self.training:
            return t
        if self.latent_noise_std <= 0.0:
            return t
        return t + self.latent_noise_std * torch.randn_like(t)

    def forward(self, batch: dict = None, **kwargs) -> dict:
        """V2's forward, with latent noise injected before each decoder call.

        Reproduces v2 logic exactly aside from the noise hooks; kept self-
        contained (rather than calling ``super().forward()``) because v2's
        masked-path block reuses the encoder internals and we want to inject
        noise at the same boundary on both paths.
        """
        if batch is None:
            batch = kwargs
        actions = batch["action"]
        actions = actions.to(dtype=self.encoder.action_proj.weight.dtype)
        device = actions.device
        B, T, D = actions.shape

        Ng = self.num_global_tokens
        Nh = self.num_hand_tokens

        # --- Encode ---
        global_tok, time_tok, hand_tok = self.encoder(actions)

        # V3: inject Gaussian noise before any decoder consumes the latents.
        global_tok_n = self._maybe_noise(global_tok)
        time_tok_n = self._maybe_noise(time_tok)
        hand_tok_n = self._maybe_noise(hand_tok)

        g_for_dec = global_tok_n if Ng > 0 else None
        h_for_dec = hand_tok_n if (Nh > 0 and self.hand_in_recon) else None

        # --- Loss 1: Recon ---
        if self.lambda_recon > 0:
            recons = self.recon_decoder(time_tok_n, global_tokens=g_for_dec, hand_tokens=h_for_dec)
            loss_recon = self._recon_loss_fn(recons, actions)
        else:
            recons = None
            loss_recon = self._zero(device)

        # --- Loss 2: Hand state prediction (uses noisy KV like decoder does) ---
        if self.lambda_hand_pred > 0 and self.hand_pred_decoder is not None:
            hand_state = batch["hand_state"].to(dtype=actions.dtype)
            future_states = batch["future_hand_states"].to(dtype=actions.dtype)
            kv_tokens = hand_tok_n if self.state_pred_kv_source == "hand" else time_tok_n
            pred_future = self.hand_pred_decoder(hand_state, kv_tokens)
            loss_hand_pred = F.mse_loss(pred_future, future_states)
        else:
            loss_hand_pred = self._zero(device)

        # --- Loss 3: Masked latent recon (+ optional masked hand pred) ---
        loss_mask_hand_pred = self._zero(device)
        if self.lambda_mask_recon > 0 and self.mask_token is not None:
            num_masked = max(1, int(B * self.mask_batch_ratio))
            perm = torch.randperm(B, device=device)[:num_masked]
            masked_actions = actions[perm].clone()

            if self.mask_ratio_min == self.mask_ratio_max:
                cur_mask_ratio = self.mask_ratio_min
            else:
                cur_mask_ratio = (
                    torch.empty(1).uniform_(self.mask_ratio_min, self.mask_ratio_max).item()
                )

            if self.mask_mode == "random":
                mask = torch.rand(num_masked, T, device=device) < cur_mask_ratio
            elif self.mask_mode == "block":
                num_mask = max(1, int(T * cur_mask_ratio))
                max_start = T - num_mask
                start = torch.randint(0, max_start + 1, (num_masked,), device=device)
                arange = torch.arange(T, device=device).unsqueeze(0)
                mask = (arange >= start.unsqueeze(1)) & (
                    arange < (start + num_mask).unsqueeze(1)
                )
            else:
                raise ValueError(f"Unknown mask_mode: {self.mask_mode}")

            masked_proj = self.encoder.action_proj(masked_actions)
            masked_proj = self.encoder.time_pos_emb(masked_proj)

            mask_expanded = mask.unsqueeze(-1).expand_as(masked_proj)
            mask_tok = self.mask_token.unsqueeze(0).unsqueeze(0).expand_as(masked_proj)
            masked_proj = torch.where(mask_expanded, mask_tok, masked_proj)

            parts = []
            if Ng > 0:
                parts.append(self.encoder.global_tokens.unsqueeze(0).expand(num_masked, -1, -1))
            parts.append(masked_proj)
            if Nh > 0:
                parts.append(self.encoder.hand_tokens.unsqueeze(0).expand(num_masked, -1, -1))
            masked_seq = torch.cat(parts, dim=1)
            masked_seq = self.encoder.transformer(masked_seq)
            # Apply V3 output postproc (output_layernorm + optional bottleneck).
            # Falls back to a single output_layernorm call for older encoders
            # that predate the encode_postproc helper.
            if hasattr(self.encoder, "encode_postproc"):
                masked_seq = self.encoder.encode_postproc(masked_seq)
            elif hasattr(self.encoder, "output_layernorm"):
                masked_seq = self.encoder.output_layernorm(masked_seq)

            m_global = masked_seq[:, :Ng] if Ng > 0 else None
            m_time = masked_seq[:, Ng:Ng + T]
            m_hand_full = masked_seq[:, Ng + T:] if Nh > 0 else None

            # V3: noise each stream once, then route the noisy tensors below.
            m_global_n = self._maybe_noise(m_global)
            m_time_n = self._maybe_noise(m_time)
            m_hand_full_n = self._maybe_noise(m_hand_full)
            m_hand_for_dec_n = m_hand_full_n if (Nh > 0 and self.hand_in_recon) else None

            recons_masked = self.recon_decoder(
                m_time_n, global_tokens=m_global_n, hand_tokens=m_hand_for_dec_n
            )
            loss_mask_recon = self._recon_loss_fn(recons_masked, actions[perm])

            if self.lambda_mask_hand_pred > 0 and self.hand_pred_decoder is not None:
                hand_state_full = batch["hand_state"].to(dtype=actions.dtype)
                future_states_gt = batch["future_hand_states"].to(dtype=actions.dtype)

                if self.state_pred_kv_source == "time":
                    m_kv = m_time_n
                elif Nh > 0:
                    # full hand chunk used for state-pred even if hand_in_recon=False
                    m_kv = m_hand_full_n
                else:
                    m_kv = None

                if m_kv is not None:
                    pred_future_masked = self.hand_pred_decoder(hand_state_full[perm], m_kv)
                    loss_mask_hand_pred = F.mse_loss(pred_future_masked, future_states_gt[perm])
        else:
            loss_mask_recon = self._zero(device)

        # --- Loss 4: Global token learning (uses CLEAN global_tok, not noisy) ---
        if (
            self.lambda_global > 0
            and self.action_text_encoder is not None
            and self.global_loss_module is not None
        ):
            fast_tokens = batch["fast_tokens"]
            text_features = self.action_text_encoder(fast_tokens)
            loss_global = self.global_loss_module(global_tok, text_features)
        else:
            loss_global = self._zero(device)

        # --- Loss 5: Frequency loss ---
        if self.freq_loss_weight > 0 and recons is not None:
            loss_freq = frequency_loss(recons, actions, loss_type=self.recon_loss_type)
        else:
            loss_freq = self._zero(device)

        loss = (
            self.lambda_recon * loss_recon
            + self.lambda_hand_pred * loss_hand_pred
            + self.lambda_mask_recon * loss_mask_recon
            + self.lambda_mask_hand_pred * loss_mask_hand_pred
            + self.lambda_global * loss_global
            + self.freq_loss_weight * loss_freq
        )

        return {
            "loss": loss,
            "loss_recon": loss_recon,
            "loss_hand_pred": loss_hand_pred,
            "loss_mask_recon": loss_mask_recon,
            "loss_mask_hand_pred": loss_mask_hand_pred,
            "loss_global": loss_global,
            "loss_freq": loss_freq,
        }
