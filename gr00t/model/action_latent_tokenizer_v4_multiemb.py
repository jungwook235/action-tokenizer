"""Multi-embodiment Action Latent Tokenizer V4 (joint training).

Trains ONE V4 (RLA-DINO hybrid) tokenizer jointly across several embodiments
that have different ``action_dim`` (e.g. GR1 vs dexjoco dual-arm). The split:

  * per-embodiment ``ActionEncoderV4`` / ``ReconDecoderV4`` (because action_dim
    differs — only the action_proj / head dims change), held in ``nn.ModuleDict``;
  * a SINGLE shared fusion encoder ``joint`` (``SimpleTokenTransformer``) and a
    SINGLE shared DINO future-feature decoder ``dino_decoder``.

There is exactly ONE ``joint`` and ONE ``dino_decoder`` in memory; every
embodiment's data flows through them and both embodiments' gradients update the
same shared weights (= joint training). No module is duplicated during training.

All embodiments must share ``action_horizon`` and the model hyper-params
(``emb_dim``, ``token_dim``, ``dino_dim``/fusion widths); only ``action_dim``
differs. A per-batch ``embodiment`` string routes to the right encoder/decoder.

Checkpoint compatibility with Stage-2: the model is saved as a SINGLE file with
each shared module stored once (no duplication). For Stage-2 (always a single
embodiment) the static helper :meth:`remap_to_single_embodiment` rewrites one
embodiment's keys into the standard single-embodiment ``ActionLatentTokenizerV4``
layout (``encoder.action_encoder.*`` / ``encoder.joint.*`` / ``recon_decoder.*``),
which the existing ``ActionLatentTokenizerWrapper`` loads unchanged.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from gr00t.model.action_latent_tokenizer_v4 import (
    _VALID_DINO_TERMS,
    ActionEncoderV4,
    ReconDecoderV4,
    _str_to_byte_tensor,
)
from gr00t.model.embod_reg import EmbodAgnosticReg
from gr00t.model.rla_modules import SimpleTokenTransformer

# [EXP-0010] Suffix of the per-domain (human) recon decoder twin. The BASE key keeps the
# plain embodiment name, so existing checkpoints and Stage-2 exports are unaffected when
# the split is off, and the twin shows up as a plain new module when it is on.
HUMAN_DECODER_SUFFIX = "__human"


class MultiEmbActionLatentTokenizerV4(nn.Module):
    """Joint multi-embodiment V4 tokenizer (shared fusion + DINO decoder).

    Args:
        embodiment_specs: list of dicts ``{"name": str, "action_dim": int}``.
            ``name`` is the routing key (no dots) and matches the Stage-2
            ``--embodiment-id``. All specs share ``action_horizon`` and the model
            hyper-params below; only ``action_dim`` varies.
        action_horizon: shared chunk length (16 for both GR1 / dexjoco).
        Other args mirror ``_build_v4_tokenizer`` in the single-embodiment script.
    """

    def __init__(
        self,
        embodiment_specs: list[dict],
        action_horizon: int,
        emb_dim: int = 256,
        head_dim: int = 64,
        encoder_depth: int = 4,
        decoder_depth: int = 2,
        decoder_mode: str = "self_attention",
        pdropout: float = 0.0,
        token_dim: int = 64,
        dino_dim: int = 1024,
        fusion_width: int = 1024,
        fusion_depth: int = 12,
        fusion_heads: int = 16,
        dino_decoder_depth: int = 12,
        use_vae: bool = False,
        vae_sample: bool = True,
        kl_free_bits: float = 0.0,
        seg_dino_decoder_depth: Optional[int] = None,
        use_seg_stream: bool = False,
        use_seg_dino_decoder: bool = False,
        lambda_recon: float = 1.0,
        lambda_dino: float = 1.0,
        lambda_dino_seg: float = 0.0,
        lambda_kl: float = 0.0,
        recon_loss_type: str = "mse",
        dino_loss_type: str = "l1",
        dino_loss_weights: Optional[dict] = None,
        feature_source: str = "dino",
        vggt_token_source: Optional[str] = None,
        vggt_image_size: Optional[int] = None,
        vggt_model: Optional[str] = None,
        vggt_final_norm: str = "none",
        dino_final_norm: str = "affine",
        use_embodiment_class_token: bool = False,
        tokenizer_finetuning_mode: bool = False,
        new_class_token: int = 0,
        num_pretrain_class_tokens: int = 0,
        embod_reg_mode: str = "",
        embod_reg_weight: float = 0.0,
        embod_reg_gather: bool = True,
        embod_reg_pool: str = "mean",
        embod_reg_vic_var: float = 1.0,
        embod_reg_vic_cov: float = 0.04,
        embod_reg_vic_std: float = 1.0,
        embod_reg_lambda: float = 1.0,
        embod_reg_human_names: Optional[list] = None,
        split_recon_decoder: bool = False,
    ):
        super().__init__()
        assert len(embodiment_specs) >= 1, "need at least one embodiment"
        self.embodiment_names = [str(s["name"]) for s in embodiment_specs]
        assert len(set(self.embodiment_names)) == len(self.embodiment_names), (
            f"duplicate embodiment names: {self.embodiment_names}"
        )
        for nm in self.embodiment_names:
            assert "." not in nm, f"embodiment name must not contain '.': {nm!r}"
        self.action_dims = {str(s["name"]): int(s["action_dim"]) for s in embodiment_specs}

        self.action_horizon = action_horizon
        self.emb_dim = emb_dim
        self.token_dim = token_dim
        self.dino_dim = dino_dim

        self.lambda_recon = float(lambda_recon)
        self.lambda_dino = float(lambda_dino)
        self.lambda_dino_seg = float(lambda_dino_seg)
        self.lambda_kl = float(lambda_kl)
        self.recon_loss_type = recon_loss_type

        # ---- segment (SAM3 cutout) DINO stream (opt-in, shared across embodiments) ----
        # When enabled the cutout stream's DINO feature difference is concatenated
        # side-by-side with the RGB difference along the token axis before the SHARED
        # fusion encoder; the concat adds no parameters (``input_layer`` is shared by
        # both streams). ``use_seg_dino_decoder`` additionally builds a SHARED twin of
        # ``dino_decoder`` that reconstructs the cutout stream's future features. Both
        # default off → state_dict / forward byte-identical to before.
        self.use_seg_stream = bool(use_seg_stream)
        self.use_seg_dino_decoder = bool(use_seg_dino_decoder)
        assert not (self.use_seg_dino_decoder and not self.use_seg_stream), (
            "use_seg_dino_decoder=True requires use_seg_stream=True (the seg features "
            "the decoder consumes come from the seg stream)."
        )
        self.use_vae = bool(use_vae)
        # VAE sampling toggle (only meaningful when use_vae). True (default) →
        # reparameterize z = μ + σ·ε (existing behavior, byte-identical path). False →
        # return the posterior mean μ (deterministic latent) while STILL computing KL, so
        # the logvar head stays in the graph (DDP-safe). Recorded as a checkpoint marker
        # (``_vae_no_sample``) below only when disabled, so the ON default keeps the
        # state_dict byte-identical to existing VAE checkpoints; Stage-2 rebuilds to match.
        self.vae_sample = bool(vae_sample)
        self.kl_free_bits = float(kl_free_bits)
        self.kl_logvar_min = -8.0
        self.kl_logvar_max = 8.0

        self.dino_terms = self._parse_dino_loss_type(dino_loss_type)
        w = {"l1": 1.0, "mse": 1.0, "cosine": 1.0}
        if dino_loss_weights:
            w.update(dino_loss_weights)
        self.dino_loss_weights = w

        # ---- per-embodiment action encoders / decoders ----
        self.action_encoders = nn.ModuleDict(
            {
                str(s["name"]): ActionEncoderV4(
                    action_dim=int(s["action_dim"]),
                    action_horizon=action_horizon,
                    emb_dim=emb_dim,
                    head_dim=head_dim,
                    depth=encoder_depth,
                    pdropout=pdropout,
                    num_global_tokens=0,
                    num_hand_tokens=0,
                )
                for s in embodiment_specs
            }
        )
        self.recon_decoders = nn.ModuleDict(
            {
                str(s["name"]): ReconDecoderV4(
                    action_dim=int(s["action_dim"]),
                    action_horizon=action_horizon,
                    emb_dim=emb_dim,
                    head_dim=head_dim,
                    depth=decoder_depth,
                    pdropout=pdropout,
                    decoder_mode=decoder_mode,
                    num_global_tokens=0,
                    num_hand_tokens=0,
                    token_dim=token_dim,
                )
                for s in embodiment_specs
            }
        )

        # ---- [EXP-0010] Change B: per-domain recon decoder split (opt-in) ----
        # ONE shared action encoder, TWO recon decoders. The base key
        # ``recon_decoders.<name>`` stays the ROBOT decoder (so a pretrained checkpoint
        # still loads strict) and the human twin is added as a NEW key
        # ``recon_decoders.<name>__human``. In finetuning-freeze mode the twin is picked
        # up automatically as a new trainable module (the "recon_decoders." prefix is
        # already on the allowed-new list). Default off -> nothing is registered.
        self.split_recon_decoder = bool(split_recon_decoder)
        self.human_decoder_suffix = HUMAN_DECODER_SUFFIX
        if self.split_recon_decoder:
            for s in embodiment_specs:
                self.recon_decoders[str(s["name"]) + HUMAN_DECODER_SUFFIX] = ReconDecoderV4(
                    action_dim=int(s["action_dim"]),
                    action_horizon=action_horizon,
                    emb_dim=emb_dim,
                    head_dim=head_dim,
                    depth=decoder_depth,
                    pdropout=pdropout,
                    decoder_mode=decoder_mode,
                    num_global_tokens=0,
                    num_hand_tokens=0,
                    token_dim=token_dim,
                )

        # ---- [EXP-0010] Change A: embodiment-agnostic regularizer on z (opt-in) ----
        # Applied to the ACTION LATENT the Stage-2 policy consumes, pooled over the time
        # axis (``mean``) or kept per time token with the token index as a stratification
        # bin (``tokens``). Default mode "" -> no module, no params, no forward change.
        self.embod_reg_mode = str(embod_reg_mode or "")
        self.embod_reg_weight = float(embod_reg_weight)
        self.embod_reg_pool = str(embod_reg_pool)
        assert self.embod_reg_pool in ("mean", "tokens"), (
            f"embod_reg_pool must be mean|tokens, got {self.embod_reg_pool!r}"
        )
        self.embod_reg_human_names = set(embod_reg_human_names or [])
        self.embod_reg = None
        if self.embod_reg_mode:
            self.embod_reg = EmbodAgnosticReg(
                d=token_dim,
                mode=self.embod_reg_mode,
                lambda_=embod_reg_lambda,
                gather=embod_reg_gather,
                vic_var=embod_reg_vic_var,
                vic_cov=embod_reg_vic_cov,
                vic_std=embod_reg_vic_std,
            )

        # ---- SHARED fusion encoder (one instance) ----
        # num_tokens=0 → action latents injected as external tokens
        # (token_channels=emb_dim); out_channels=token_dim → out_layer IS the
        # bottleneck. Identical config to TimeWiseEncoderV4.joint.
        self.joint = SimpleTokenTransformer(
            in_channels=dino_dim,
            model_channels=fusion_width,
            out_channels=token_dim,
            num_blocks=fusion_depth,
            num_heads=fusion_heads,
            num_tokens=0,
            token_channels=emb_dim,
            use_fp16=False,
        )

        # ---- SHARED VAE logvar head (one instance; token_dim→token_dim, embodiment-agnostic) ----
        if self.use_vae:
            self.logvar_head = nn.Linear(token_dim, token_dim)
            nn.init.zeros_(self.logvar_head.weight)
            nn.init.constant_(self.logvar_head.bias, -5.0)

        # ---- SHARED DINO future-feature decoder (one instance) ----
        self.dino_decoder = SimpleTokenTransformer(
            in_channels=dino_dim,
            model_channels=fusion_width,
            out_channels=dino_dim,
            num_blocks=dino_decoder_depth,
            num_heads=fusion_heads,
            num_tokens=action_horizon,
            token_channels=token_dim,
            zero_init=True,
            use_fp16=False,
        )

        # ---- SHARED segment-stream DINO decoder (opt-in; identical config) ----
        self.seg_dino_decoder = None
        if self.use_seg_dino_decoder:
            self.seg_dino_decoder = SimpleTokenTransformer(
                in_channels=dino_dim,
                model_channels=fusion_width,
                out_channels=dino_dim,
                num_blocks=int(seg_dino_decoder_depth or dino_decoder_depth),
                num_heads=fusion_heads,
                num_tokens=action_horizon,
                token_channels=token_dim,
                zero_init=True,
                use_fp16=False,
            )

        # ---- per-embodiment (data-type) learnable class token (opt-in) ----
        # When enabled, ONE learnable [dino_dim] vector per class-token id is prepended
        # to the DINO feature sequence entering both the shared fusion encoder and the
        # shared DINO decoder, so those shared modules can condition on the data type.
        # Which token a group uses is declared explicitly by ``class_token_id`` in the
        # embodiments JSON (groups may share a token by reusing an id). When disabled
        # (default) NO params/buffers are registered → state_dict is byte-identical to
        # before and the forward path is unchanged.
        # ---- tokenizer finetuning mode (opt-in; adds a NEW embodiment to a
        # pretrained joint tokenizer) ----
        # When enabled the per-embodiment action encoder/decoder for a new embodiment
        # (built above from ``embodiment_specs``) are trained on top of a loaded
        # pretrained checkpoint (the shared fusion + DINO decoder + old embodiments load
        # via ``load_state_dict(strict=False)``; the new enc/dec stay randomly-init'd).
        # ``new_class_token`` (>0) adds that many NEW learnable class tokens in a SEPARATE
        # ``finetuning_class_token`` parameter (so the base ``embodiment_class_token`` keeps
        # its pretrained name/shape and loads strict, and a freeze can target only the new
        # rows). ``num_pretrain_class_tokens`` (= rows in the pretrained
        # ``embodiment_class_token``) is the boundary between base and finetuning ids.
        # Default (False/0) is byte-identical to the original construction below.
        self.tokenizer_finetuning_mode = bool(tokenizer_finetuning_mode)
        self.new_class_token = int(new_class_token)
        self.num_pretrain_class_tokens = int(num_pretrain_class_tokens)
        assert self.new_class_token >= 0, "new_class_token must be >= 0"
        if self.new_class_token > 0:
            assert use_embodiment_class_token, (
                "new_class_token > 0 requires use_embodiment_class_token=True "
                "(new class tokens ARE embodiment class tokens)."
            )
            assert self.tokenizer_finetuning_mode, (
                "new_class_token > 0 is only valid in tokenizer_finetuning_mode."
            )

        self.use_embodiment_class_token = bool(use_embodiment_class_token)
        if self.use_embodiment_class_token:
            class_token_ids = {}
            for s in embodiment_specs:
                nm = str(s["name"])
                assert "class_token_id" in s, (
                    f"[{nm}] class_token_id is required in every embodiment spec when "
                    f"use_embodiment_class_token=True (set it in the embodiments JSON)."
                )
                cid = int(s["class_token_id"])
                assert cid >= 0, f"[{nm}] class_token_id must be >= 0; got {cid}"
                class_token_ids[nm] = cid
            self.class_token_ids = class_token_ids

            if self.tokenizer_finetuning_mode:
                # Base param sized to the pretrained token count so its key loads strict;
                # new ids (>= base) live in a separate learnable ``finetuning_class_token``.
                # base_n == 0 means the pretrained tokenizer had NO class tokens: there is no
                # base param at all and EVERY class token is new (prompt-tuning-style — the
                # frozen fusion/DINO-decoder learn to attend to a brand-new learnable token).
                base_n = self.num_pretrain_class_tokens
                assert base_n >= 0, "num_pretrain_class_tokens must be >= 0"
                total = base_n + self.new_class_token
                assert total >= 1, (
                    "tokenizer_finetuning_mode with class tokens needs >= 1 total token "
                    "(num_pretrain_class_tokens + new_class_token)."
                )
                max_cid = max(class_token_ids.values())
                assert max_cid < total, (
                    f"class_token_id {max_cid} out of range: base={base_n} + "
                    f"new_class_token={self.new_class_token} = {total} slots."
                )
                for nm, cid in class_token_ids.items():
                    if cid >= base_n:
                        assert self.new_class_token > 0, (
                            f"[{nm}] class_token_id={cid} >= num_pretrain_class_tokens="
                            f"{base_n} requires new_class_token > 0."
                        )
                # Base param only when the pretrained ckpt actually had class tokens; it then
                # loads strict against them. base_n == 0 → skip it entirely (no empty param).
                if base_n > 0:
                    self.embodiment_class_token = nn.Parameter(
                        torch.randn(base_n, dino_dim) * 0.02
                    )
                if self.new_class_token > 0:
                    self.finetuning_class_token = nn.Parameter(
                        torch.randn(self.new_class_token, dino_dim) * 0.02
                    )
            else:
                num_class_tokens = max(class_token_ids.values()) + 1
                self.embodiment_class_token = nn.Parameter(
                    torch.randn(num_class_tokens, dino_dim) * 0.02
                )
            # Buffers so remap_to_single_embodiment (which only sees the state_dict) can
            # resolve name → id and slice out the right row for Stage-2.
            self.register_buffer("_use_emb_class_token", torch.tensor(True))
            for nm, cid in class_token_ids.items():
                self.register_buffer(f"_class_token_id__{nm}", torch.tensor(cid))

        # ---- detection / metadata buffers ----
        # _is_v4_multiemb routes ActionLatentTokenizerWrapper.from_checkpoint to the
        # remap path. The single-embodiment marker _is_v4 is added by remap, NOT here.
        self.register_buffer("_is_v4_multiemb", torch.tensor(True))
        # Segment-stream marker: registered ONLY when enabled, so ordinary checkpoints
        # stay byte-identical. Carried through remap so Stage-2 rebuilds an encoder that
        # expects the seg features.
        if self.use_seg_stream:
            self.register_buffer("_use_seg_stream", torch.tensor(True))
        if self.use_vae:
            self.register_buffer("_is_vae", torch.tensor(True))
            # Sampling-off marker: registered ONLY when a VAE tokenizer disables sampling
            # (encode returns μ). Absence ⇒ sampling ON (default), so ordinary VAE
            # checkpoints stay byte-identical. Carried through remap so Stage-2 rebuilds
            # the matching (deterministic-μ) encoder.
            if not self.vae_sample:
                self.register_buffer("_vae_no_sample", torch.tensor(True))

        self.feature_source = feature_source
        if feature_source == "vggt":
            assert vggt_token_source in ("aggregator", "dpt_out2"), (
                f"vggt_token_source must be 'aggregator' or 'dpt_out2'; got {vggt_token_source!r}"
            )
            self.register_buffer("_feature_source", _str_to_byte_tensor("vggt"))
            self.register_buffer("_vggt_token_source", _str_to_byte_tensor(str(vggt_token_source)))
            self.register_buffer("_vggt_image_size", torch.tensor(int(vggt_image_size or 224)))
            self.register_buffer("_vggt_model", _str_to_byte_tensor(str(vggt_model or "facebook/VGGT-1B")))
            if vggt_final_norm == "naive":
                self.register_buffer("_vggt_final_norm", _str_to_byte_tensor("naive"))
        self.vggt_final_norm = vggt_final_norm

        self.dino_final_norm = dino_final_norm
        if feature_source == "dino" and dino_final_norm == "naive":
            self.register_buffer("_dino_final_norm", _str_to_byte_tensor("naive"))

    # ---- loss helpers (copied from ActionLatentTokenizerV4) ----

    @staticmethod
    def _parse_dino_loss_type(s: str):
        terms = [t.strip().lower() for t in str(s).split("+") if t.strip()]
        if not terms:
            raise ValueError(f"Empty dino_loss_type: {s!r}")
        for t in terms:
            if t not in _VALID_DINO_TERMS:
                raise ValueError(f"Unknown dino loss term {t!r}; valid: {_VALID_DINO_TERMS}")
        return terms

    def _recon_loss_fn(self, pred, target):
        if self.recon_loss_type == "l1":
            return F.l1_loss(pred, target)
        return F.mse_loss(pred, target)

    def _dino_loss(self, pred: torch.Tensor, target: torch.Tensor):
        sub = {}
        total = pred.new_zeros(())
        target = target.to(dtype=pred.dtype)
        for t in self.dino_terms:
            if t == "l1":
                v = F.l1_loss(pred, target)
            elif t == "mse":
                v = F.mse_loss(pred, target)
            else:  # cosine
                v = (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean()
            sub[f"loss_dino_{t}"] = v
            total = total + self.dino_loss_weights[t] * v
        return total, sub

    # ---- encode / decode (single embodiment per call) ----

    def _class_token_row(self, name: str) -> torch.Tensor:
        """Return ``name``'s [dino_dim] class-token row.

        Finetuning ids at/above ``num_pretrain_class_tokens`` come from the separate
        ``finetuning_class_token`` parameter; everything else from the base
        ``embodiment_class_token``. Off the finetuning path this is exactly
        ``embodiment_class_token[cid]``."""
        cid = self.class_token_ids[name]
        if (
            self.tokenizer_finetuning_mode
            and self.new_class_token > 0
            and cid >= self.num_pretrain_class_tokens
        ):
            return self.finetuning_class_token[cid - self.num_pretrain_class_tokens]
        return self.embodiment_class_token[cid]

    def _prepend_class_token(self, name: str, x: torch.Tensor) -> torch.Tensor:
        """Prepend ``name``'s learnable data-type class token as an extra
        ``dino_dim``-channel patch: [B,Lp,dino_dim] → [B,1+Lp,dino_dim]."""
        ct = self._class_token_row(name).to(dtype=x.dtype)       # [dino_dim]
        ct = ct.view(1, 1, -1).expand(x.shape[0], 1, -1)          # [B,1,dino_dim]
        return torch.cat([ct, x], dim=1)

    def encode(
        self,
        name: str,
        actions: torch.Tensor,
        x0_feat: torch.Tensor,
        x1_feat: torch.Tensor,
        s0_feat: Optional[torch.Tensor] = None,
        s1_feat: Optional[torch.Tensor] = None,
    ):
        """Route ``name``'s action encoder → shared fusion → time latent [B,T,token_dim].

        With the segment stream enabled, ``s0_feat``/``s1_feat`` (cutout-frame DINO
        features) contribute a second feature-difference sequence concatenated
        side-by-side with the RGB one.

        Returns ``(time_tok, kl)`` where ``kl`` is a scalar tensor (VAE) or None.
        """
        action_encoder = self.action_encoders[name]
        actions = actions.to(dtype=action_encoder.action_proj.weight.dtype)
        x0_feat = x0_feat.to(dtype=actions.dtype)
        x1_feat = x1_feat.to(dtype=actions.dtype)

        _, t256, _ = action_encoder(actions)               # [B,T,emb_dim]
        dino_diff = x1_feat - x0_feat                       # [B,Lp,dino_dim]
        if self.use_seg_stream:
            assert s0_feat is not None and s1_feat is not None, (
                "use_seg_stream=True; encode() requires s0_feat/s1_feat."
            )
            seg_diff = s1_feat.to(dtype=actions.dtype) - s0_feat.to(dtype=actions.dtype)
            # Side-by-side concat along the token axis: [B, Lp + Lp_seg, dino_dim].
            dino_diff = torch.cat([dino_diff, seg_diff], dim=1)
        if self.use_embodiment_class_token:
            # Extra class-token patch lands in the discarded (visual) half of the
            # fusion output; the kept action-token positions are unchanged in shape.
            dino_diff = self._prepend_class_token(name, dino_diff)  # [B,1+Lp,dino_dim]
        tokens_out, _ = self.joint(x=dino_diff, tokens=t256)  # [B,T,token_dim]

        kl = None
        if self.use_vae:
            # SD-style VAE: fusion output = posterior mean μ. vae_sample True (default)
            # reparameterizes z = μ + σ·ε; False returns μ (deterministic). KL is computed
            # either way, so the logvar head stays trained / in-graph (DDP-safe).
            mu = tokens_out
            logvar = self.logvar_head(mu).clamp(self.kl_logvar_min, self.kl_logvar_max)
            if self.vae_sample:
                std = torch.exp(0.5 * logvar)
                tokens_out = mu + torch.randn_like(std) * std
            else:
                tokens_out = mu
            kl_dim = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())  # [B,T,token_dim]
            kl_dim = kl_dim.mean(dim=(0, 1))                           # [token_dim]
            if self.kl_free_bits > 0:
                kl_dim = torch.clamp(kl_dim, min=self.kl_free_bits)
            kl = kl_dim.sum()
        return tokens_out, kl

    def decode(self, name: str, time_tok: torch.Tensor) -> torch.Tensor:
        return self.recon_decoders[name](time_tok)

    def _recon_loss(self, name, time_tok, actions, is_human):
        """Reconstruction loss, optionally routed to per-domain decoders (Change B).

        Off (default) or without labels: the single ``recon_decoders[name]`` -- the
        original path. On: robot rows decode through ``recon_decoders[name]``, human rows
        through the ``__human`` twin, and the two subset means are recombined weighted by
        row count. That weighting reproduces the plain mean EXACTLY, so with the twin
        copy-initialized from the base decoder, enabling the split cannot by itself move
        the loss -- the two only diverge as their own domain's gradients arrive.
        """
        if not self.split_recon_decoder or is_human is None:
            return self._recon_loss_fn(self.decode(name, time_tok), actions)
        hm = is_human.reshape(-1).to(time_tok.device) > 0.5
        n = int(hm.numel())
        n_h = int(hm.sum().item())
        human_name = name + self.human_decoder_suffix
        if n_h == 0:
            return self._recon_loss_fn(self.decode(name, time_tok), actions)
        if n_h == n:
            return self._recon_loss_fn(self.decode(human_name, time_tok), actions)
        l_r = self._recon_loss_fn(self.decode(name, time_tok[~hm]), actions[~hm])
        l_h = self._recon_loss_fn(self.decode(human_name, time_tok[hm]), actions[hm])
        return (float(n - n_h) * l_r + float(n_h) * l_h) / float(n)

    def _reg_labels(self, name, is_human, B, device):
        """Per-sample domain label for the regularizer.

        Prefers the batch's own ``is_human``: in the {p,r,q} config robot and human
        sources are MIXED inside one embodiment group, so the group name alone cannot
        tell them apart. Falls back to group membership (``embod_reg_human_names``) for
        configs where a whole embodiment is one domain.
        """
        if is_human is not None:
            return is_human.reshape(-1).float().to(device)
        return torch.full((B,), float(name in self.embod_reg_human_names),
                          device=device, dtype=torch.float32)

    def _pool_for_reg(self, time_tok, name, is_human):
        """(h, labels, bin_ids) for the regularizer, from the latent z [B,T,token_dim].

        ``mean``   -> [B, token_dim]; isomorphic to the reference implementation's
                      pooled hidden, and the safe default.
        ``tokens`` -> [B*T, token_dim] with ``bin_ids`` = the time-token index, so the
                      invariance term contrasts human vs robot AT THE SAME chunk phase
                      ("same phase -> same latent") instead of only matching marginals.
        """
        B, T, _ = time_tok.shape
        lbl = self._reg_labels(name, is_human, B, time_tok.device)
        if self.embod_reg_pool == "tokens":
            h = time_tok.reshape(B * T, -1)                       # index = b*T + t
            bins = torch.arange(T, device=time_tok.device).repeat(B)
            return h, lbl.repeat_interleave(T), bins
        return time_tok.mean(dim=1), lbl, None

    def _apply_reg(self, out: dict, reg_collect) -> dict:
        """Add the regularizer once over ALL groups' latents (one contrast per step).

        Pooling across groups first is also what makes ``gather=True`` safe: N is then
        the full per-rank micro-batch, identical on every rank.
        """
        if self.embod_reg is None or not reg_collect:
            return out
        h = torch.cat([c[0] for c in reg_collect], 0)
        lbl = torch.cat([c[1] for c in reg_collect], 0)
        bins = (torch.cat([c[2] for c in reg_collect], 0)
                if reg_collect[0][2] is not None else None)
        reg = self.embod_reg(h, lbl, bins)
        out["loss"] = out["loss"] + self.embod_reg_weight * reg
        dev = h.device
        # Scalars only (the trainer .item()s every non-"loss" entry). The two counts are
        # the proof that the domain labels actually arrived -- they are what the contrast
        # was computed on, after the all-gather when gather=True.
        out["loss_embod_reg"] = reg.detach()
        out["embod_reg_n_human"] = torch.tensor(self.embod_reg.last_nh, device=dev)
        out["embod_reg_n_robot"] = torch.tensor(self.embod_reg.last_nr, device=dev)
        out["embod_reg_gap"] = torch.tensor(self.embod_reg.last_gap, device=dev)
        out["embod_reg_std_min"] = torch.tensor(self.embod_reg.last_std_min, device=dev)
        if bins is not None:
            out["embod_reg_bins"] = torch.tensor(self.embod_reg.last_bins, device=dev)
        return out

    def decode_dino(self, time_tok: torch.Tensor, x0_feat: torch.Tensor,
                    name: Optional[str] = None) -> torch.Tensor:
        if self.use_embodiment_class_token:
            assert name is not None, "decode_dino needs `name` when use_embodiment_class_token=True"
            x0_feat = self._prepend_class_token(name, x0_feat)   # [B,1+Lp,dino_dim]
        _, visuals = self.dino_decoder(x=x0_feat, tokens=time_tok)
        if self.use_embodiment_class_token:
            visuals = visuals[:, 1:]                              # drop class-token slot
        return visuals

    def decode_dino_seg(self, time_tok: torch.Tensor, s0_feat: torch.Tensor,
                        name: Optional[str] = None) -> torch.Tensor:
        """Segment-stream twin of :meth:`decode_dino` (predicts s1_feat from s0_feat)."""
        assert self.seg_dino_decoder is not None, "seg_dino_decoder was not built"
        if self.use_embodiment_class_token:
            assert name is not None, (
                "decode_dino_seg needs `name` when use_embodiment_class_token=True"
            )
            s0_feat = self._prepend_class_token(name, s0_feat)   # [B,1+Lp,dino_dim]
        _, visuals = self.seg_dino_decoder(x=s0_feat, tokens=time_tok)
        if self.use_embodiment_class_token:
            visuals = visuals[:, 1:]                              # drop class-token slot
        return visuals

    def forward(self, batch: dict = None, **kwargs) -> dict:
        """Two call shapes:

        * single embodiment: ``{embodiment, action, x0_feat, x1_feat}``.
        * multiple groups (one forward per training step, DDP-safe):
          ``{"groups": {name: {action, x0_feat, x1_feat}}, "embodiment_order": [...]}``.
          Per-group losses are summed weighted by group batch size so the total
          matches a single-embodiment run's scale; per-embodiment components are
          tagged ``<key>/<name>`` for logging.
        """
        if batch is None:
            batch = kwargs
        if "groups" in batch:
            groups = batch["groups"]
            order = batch.get("embodiment_order") or list(groups.keys())
            total_n = sum(int(groups[n]["action"].shape[0]) for n in order)
            total_n = max(total_n, 1)
            agg: dict = {}
            loss = None
            # [EXP-0010] one regularizer contrast per step, over every group's latents.
            reg_collect = [] if self.embod_reg is not None else None
            for name in order:
                g = groups[name]
                out = self._forward_single(
                    name, g["action"], g["x0_feat"], g["x1_feat"],
                    g.get("s0_feat"), g.get("s1_feat"),
                    is_human=g.get("is_human"), reg_collect=reg_collect,
                )
                w = int(g["action"].shape[0]) / total_n
                loss = out["loss"] * w if loss is None else loss + out["loss"] * w
                for k, v in out.items():
                    if k != "loss":
                        agg[f"{k}/{name}"] = v
            agg["loss"] = loss
            return self._apply_reg(agg, reg_collect)
        reg_collect = [] if self.embod_reg is not None else None
        out = self._forward_single(
            batch["embodiment"], batch["action"], batch["x0_feat"], batch["x1_feat"],
            batch.get("s0_feat"), batch.get("s1_feat"),
            is_human=batch.get("is_human"), reg_collect=reg_collect,
        )
        return self._apply_reg(out, reg_collect)

    def _forward_single(self, name, actions, x0_feat, x1_feat,
                        s0_feat=None, s1_feat=None,
                        is_human=None, reg_collect=None) -> dict:
        if isinstance(name, (list, tuple)):
            name = name[0]
        device = actions.device

        time_tok, kl = self.encode(name, actions, x0_feat, x1_feat, s0_feat, s1_feat)
        actions = actions.to(dtype=time_tok.dtype)

        # [EXP-0010] hand the latent to the caller for one pooled contrast per step.
        if reg_collect is not None:
            reg_collect.append(self._pool_for_reg(time_tok, name, is_human))

        if self.lambda_recon > 0:
            loss_recon = self._recon_loss(name, time_tok, actions, is_human)
        else:
            loss_recon = torch.zeros((), device=device)

        if self.lambda_dino > 0:
            pred_x1 = self.decode_dino(time_tok, x0_feat.to(dtype=time_tok.dtype), name=name)
            loss_dino, dino_sub = self._dino_loss(pred_x1, x1_feat)
        else:
            loss_dino = torch.zeros((), device=device)
            dino_sub = {}

        loss = self.lambda_recon * loss_recon + self.lambda_dino * loss_dino

        # Segment-stream DINO future-feature reconstruction (same mechanics as the RGB
        # DINO loss, on the cutout stream). Skipped entirely when the decoder is absent.
        out_seg = {}
        if self.seg_dino_decoder is not None:
            assert s0_feat is not None and s1_feat is not None, (
                "seg_dino_decoder is present but the batch has no s0_feat/s1_feat."
            )
            pred_s1 = self.decode_dino_seg(
                time_tok, s0_feat.to(dtype=time_tok.dtype), name=name
            )
            loss_dino_seg, seg_sub = self._dino_loss(pred_s1, s1_feat)
            loss = loss + self.lambda_dino_seg * loss_dino_seg
            out_seg["loss_dino_seg"] = loss_dino_seg
            out_seg.update({f"{k}_seg": v for k, v in seg_sub.items()})

        out = {"loss": loss, "loss_recon": loss_recon, "loss_dino": loss_dino}
        out.update(out_seg)
        if self.use_vae and kl is not None:
            if self.lambda_kl > 0:
                loss = loss + self.lambda_kl * kl
            out["loss"] = loss
            out["loss_kl"] = kl
        out.update(dino_sub)
        return out

    # ---- Stage-2 / inference: extract one embodiment in standard v4 layout ----

    @staticmethod
    def remap_to_single_embodiment(state_dict: dict, name: str) -> dict:
        """Rewrite a multi-embodiment state_dict into the standard single-embodiment
        ``ActionLatentTokenizerV4`` key layout for ``name``.

        Maps ``action_encoders.<name>.*`` → ``encoder.action_encoder.*``,
        ``recon_decoders.<name>.*`` → ``recon_decoder.*``, shared ``joint.*`` →
        ``encoder.joint.*`` and ``logvar_head.*`` → ``encoder.logvar_head.*``.
        Drops ``dino_decoder.*`` / ``seg_dino_decoder.*`` (Stage-2 wrapper builds them
        as None). Adds the ``_is_v4`` marker so ``_build_from_state_dict`` routes to the
        v4 builder. The ``_use_seg_stream`` marker carries over so Stage-2 rebuilds an
        encoder that expects the segment-stream features.

        When the joint tokenizer was trained with per-embodiment class tokens
        (``_use_emb_class_token`` present), slices ``embodiment_class_token`` down to
        this embodiment's single row (via ``_class_token_id__<name>``) and emits it as
        ``encoder.embodiment_class_token`` so Stage-2's ``TimeWiseEncoderV4`` prepends
        the same token. The multiemb-only class-token keys are dropped.
        """
        # [EXP-0010] ``name`` may carry the per-domain decoder suffix (e.g.
        # "openarm_prq__human"). The action ENCODER and the class token are shared, so
        # they come from the BASE embodiment; only the recon DECODER is the twin. That is
        # how Stage-2 selects "decode this latent the human way" vs "the robot way" --
        # same encoder, same latent space, different output style.
        base = name
        if HUMAN_DECODER_SUFFIX in name and not any(
            k.startswith(f"action_encoders.{name}.") for k in state_dict
        ):
            base = name[: name.rindex(HUMAN_DECODER_SUFFIX)]
        pfx_ae = f"action_encoders.{base}."
        pfx_rd = f"recon_decoders.{name}."

        # Per-embodiment class token → single-embodiment encoder buffer.
        # Finetuning checkpoints keep the pretrained ``embodiment_class_token`` (base rows)
        # plus a separate ``finetuning_class_token`` (new rows). Ids at/above the base row
        # count resolve into the finetuning parameter; the emitted single row is identical
        # in shape/meaning either way, so Stage-2 stays unchanged.
        emb_class_token = None
        if "_use_emb_class_token" in state_dict:
            id_key = f"_class_token_id__{base}"
            assert id_key in state_dict, (
                f"class-token checkpoint missing {id_key!r} for embodiment {name!r}"
            )
            cid = int(state_dict[id_key].item())
            base_n = (state_dict["embodiment_class_token"].shape[0]
                      if "embodiment_class_token" in state_dict else 0)
            if "finetuning_class_token" in state_dict and cid >= base_n:
                emb_class_token = state_dict["finetuning_class_token"][cid - base_n].clone()
            else:
                emb_class_token = state_dict["embodiment_class_token"][cid].clone()

        out: dict = {}
        found = False
        for k, v in state_dict.items():
            if k.startswith(pfx_ae):
                out["encoder.action_encoder." + k[len(pfx_ae):]] = v
                found = True
            elif k.startswith(pfx_rd):
                out["recon_decoder." + k[len(pfx_rd):]] = v
            elif k.startswith("joint."):
                out["encoder.joint." + k[len("joint."):]] = v
            elif k.startswith("logvar_head."):
                out["encoder.logvar_head." + k[len("logvar_head."):]] = v
            elif k.startswith("action_encoders.") or k.startswith("recon_decoders."):
                continue  # other embodiment
            elif k.startswith("dino_decoder.") or k.startswith("seg_dino_decoder."):
                continue  # training-only; Stage-2 wrapper uses (seg_)dino_decoder=None
            elif k == "_is_v4_multiemb":
                continue  # replaced by _is_v4 below
            elif k in ("_is_vae", "_vae_no_sample", "_use_seg_stream", "_dino_final_norm",
                       "_feature_source", "_vggt_token_source", "_vggt_image_size",
                       "_vggt_model", "_vggt_final_norm"):
                out[k] = v  # top-level detection markers carry over unchanged
        if not found:
            available = sorted(
                k.split(".")[1] for k in state_dict if k.startswith("action_encoders.")
            )
            raise ValueError(
                f"embodiment_id {name!r} not found in checkpoint; "
                f"available: {sorted(set(available))}"
            )
        if emb_class_token is not None:
            out["encoder.embodiment_class_token"] = emb_class_token
        out["_is_v4"] = torch.tensor(True)
        return out
