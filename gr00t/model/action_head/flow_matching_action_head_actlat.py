# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
import math
from torch.nn import init

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature
from typing import Optional
from gr00t.model.action_head.action_encoder import (
    SinusoidalPositionalEncoding,
    swish,
)

from .cross_attention_dit import DiT, SelfAttentionTransformer


class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError(
                "Expected `timesteps` to have shape (B,) so we can replicate across T."
            )

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.W1(actions, cat_ids)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # 5) Finally W3 => (B, T, w)
        x = self.W3(x, cat_ids)
        return x


@dataclass
class FlowmatchingActionHeadConfig(PretrainedConfig):
    """NOTE: N1.5 uses XEmbFlowmatchingPolicyHeadConfig as action head"""

    add_pos_embed: bool = field(
        default=True, metadata={"help": "Whether to add positional embedding"}
    )
    model_dtype: str = field(default="float32", metadata={"help": "Model data type."})
    diffusion_model_cfg: dict = field(
        default=None, metadata={"help": "Diffusion model configuration."}
    )
    input_embedding_dim: int = field(
        default=1536, metadata={"help": "Input embedding channel dimension."}
    )
    backbone_embedding_dim: int = field(
        default=1536, metadata={"help": "Backbone embedding channel dimension."}
    )

    hidden_size: int = field(default=1024, metadata={"help": "Input embedding dimension."})
    max_seq_len: int = field(default=1024, metadata={"help": "Maxium Sequence Length"})
    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})
    noise_beta_alpha: float = field(default=1.5, metadata={"help": ""})
    noise_beta_beta: float = field(default=1.0, metadata={"help": ""})
    noise_s: float = field(
        default=0.999, metadata={"help": "Flow matching noise Beta distribution s."}
    )
    num_timestep_buckets: int = field(
        default=1000, metadata={"help": "Number of timestep discretization buckets."}
    )
    num_inference_timesteps: int = field(
        default=None,
        metadata={"help": "Number of inference steps for noise diffusion."},
    )
    max_num_embodiments: int = field(default=32, metadata={"help": "Number of embodiments."})
    tune_projector: bool = field(default=True, metadata={"help": "Whether to tune the projector."})
    tune_diffusion_model: bool = field(
        default=True, metadata={"help": "Whether to tune the diffusion model."}
    )
    load_pretrained_det_decode_layer_path: str = field(
        default=None, metadata={"help": "Path to pretrained detection model."}
    )
    detection_coeff: float = field(default=1.0, metadata={"help": "Detection coefficient."})

    freeze_decode_layer: bool = field(default=False)
    expand_batch: int = field(default=None)
    use_vlln: bool = field(default=True)

    vl_self_attention_cfg: dict = field(default=None)
    # Kept for checkpoint compatibility (future_tokens declaration)
    num_target_vision_tokens: int = field(
        default=32, metadata={"help": "Number of target vision tokens (kept for compat)."}
    )
    flare_loss_lambda: float = field(default=0.0, metadata={"help": "Unused, kept for compat."})
    flare_align_layers: int = field(default=12, metadata={"help": "Unused, kept for compat."})

    image_count: int = field(default=1, metadata={"help": "Unused, kept for compat."})
    video_only: bool = field(default=False, metadata={"help": "Whether to only use video."})
    flare_image_time: str = field(default="future", metadata={"help": "Unused, kept for compat."})

    # Action latent prediction config
    actlat_loss_lambda: float = field(
        default=0.0, metadata={"help": "Lambda for action latent prediction loss."}
    )
    actlat_num_tokens: int = field(
        default=0, metadata={"help": "Number of learnable action-latent tokens in DiT."}
    )
    actlat_latent_dim: int = field(
        default=256, metadata={"help": "Embedding dim of the action latent encoder."}
    )
    actlat_align_layers: int = field(
        default=12, metadata={"help": "DiT layer index to extract hidden states from."}
    )
    actlat_loss_type: str = field(
        default="mse",
        metadata={"help": "Loss type for action latent prediction: 'mse' or 'cosine'."},
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class FlowmatchingActionHead(nn.Module):
    config_class = FlowmatchingActionHeadConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: FlowmatchingActionHeadConfig,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        print("config.diffusion_model_cfg", config.diffusion_model_cfg)

        self.model = DiT(**config.diffusion_model_cfg)
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=config.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        # Kept for checkpoint compatibility — not used in forward/get_action
        if config.num_target_vision_tokens > 0:
            self.future_tokens = nn.Embedding(
                config.num_target_vision_tokens, self.input_embedding_dim
            )
            nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)
            self.siglip_embedding_size = 1152
            self.flare_projector = nn.Sequential(
                nn.Linear(self.input_embedding_dim, self.siglip_embedding_size),
                nn.SiLU(),
                nn.Linear(self.siglip_embedding_size, self.siglip_embedding_size),
                nn.SiLU(),
                nn.Linear(self.siglip_embedding_size, self.siglip_embedding_size),
            )

        # Action latent prediction modules
        if config.actlat_num_tokens > 0:
            self.actlat_tokens = nn.Embedding(
                config.actlat_num_tokens, self.input_embedding_dim
            )
            nn.init.normal_(self.actlat_tokens.weight, mean=0.0, std=0.02)
            self.actlat_projector = nn.Sequential(
                nn.Linear(self.input_embedding_dim, self.input_embedding_dim),
                nn.SiLU(),
                nn.Linear(self.input_embedding_dim, config.actlat_latent_dim),
                nn.SiLU(),
                nn.Linear(config.actlat_latent_dim, config.actlat_latent_dim),
            )

        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        )
        self.vl_self_attention = (
            SelfAttentionTransformer(**config.vl_self_attention_cfg)
            if config.use_vlln
            else nn.Identity()
        )

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.config = config
        self.set_trainable_parameters(config.tune_projector, config.tune_diffusion_model)

    def set_trainable_parameters(self, tune_projector: bool, tune_diffusion_model: bool):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        # Future tokens / flare projector are kept frozen (compat only)
        if hasattr(self, "future_tokens"):
            self.future_tokens.requires_grad_(False)
        if hasattr(self, "flare_projector"):
            self.flare_projector.requires_grad_(False)
        print(f"Tune action head projector: {self.tune_projector}")
        print(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        if not tune_projector and not tune_diffusion_model:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No action head trainable parameters found.")

    def _init_weights(self, module, name):
        if "weight" in name:
            init.kaiming_uniform_(module, a=math.sqrt(5))
        if "bias" in name:
            weight_name = name.replace("bias", "weight")
            weight = self.get_parameter(weight_name)

            fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(module, -bound, bound)

    def set_frozen_modules_to_eval_mode(self):
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        backbone_features = self.vl_self_attention(backbone_features)
        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def forward(
        self, backbone_output: BatchFeature, action_input: BatchFeature
    ) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()
        backbone_output = self.process_backbone_output(backbone_output)

        if self.config.expand_batch is not None:
            for k, v in backbone_output.items():
                ndim = len(v.shape)
                factors = tuple([self.config.expand_batch] + [1] * (ndim - 1))
                backbone_output[k] = v.repeat(*factors)
            for k, v in action_input.items():
                if isinstance(v, torch.Tensor):
                    ndim = len(v.shape)
                    factors = tuple([self.config.expand_batch] + [1] * (ndim - 1))
                    action_input[k] = v.repeat(*factors)

        vl_embs = backbone_output.backbone_features
        device = vl_embs.device
        embodiment_id = action_input.embodiment_id

        # Embed state
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Embed noised action trajectory
        actions = action_input.action
        B = actions.shape[0]
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]

        noisy_trajectory = (1 - t) * noise + t * actions
        if self.config.video_only:
            noisy_trajectory = noise

        velocity = actions - noise

        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Build sequence: [state | actlat_tokens | action]
        parts = [state_features]
        actlat_start = state_features.shape[1]  # where actlat tokens start in sequence
        if self.config.actlat_num_tokens > 0:
            al_tokens = self.actlat_tokens.weight.unsqueeze(0).expand(B, -1, -1)
            parts.append(al_tokens)
        actlat_end = actlat_start + self.config.actlat_num_tokens
        parts.append(action_features)
        sa_embs = torch.cat(parts, dim=1)

        vl_attn_mask = backbone_output.backbone_attention_mask

        need_hidden = self.config.actlat_num_tokens > 0 and self.config.actlat_loss_lambda > 0
        if need_hidden:
            model_output, model_hidden_states = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
            )
        else:
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
            )

        # Action loss
        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1]:]

        action_mask = action_input.action_mask
        loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        loss = loss.sum() / action_mask.sum()

        if self.config.video_only:
            loss *= 0.0

        # Action latent prediction loss
        actlat_loss = 0.0
        if need_hidden:
            hidden = model_hidden_states[self.config.actlat_align_layers]
            actlat_hidden = hidden[:, actlat_start:actlat_end, :]
            actlat_pred = self.actlat_projector(actlat_hidden)
            actlat_target = action_input["actlat_target"]

            if self.config.actlat_loss_type == "mse":
                actlat_loss = F.mse_loss(actlat_pred, actlat_target)
            elif self.config.actlat_loss_type == "cosine":
                actlat_loss = -F.cosine_similarity(actlat_pred, actlat_target, dim=-1).mean()
            else:
                raise ValueError(f"Unknown actlat_loss_type: {self.config.actlat_loss_type}")

            actlat_loss = self.config.actlat_loss_lambda * actlat_loss

        loss = loss + actlat_loss

        output_dict = {"loss": loss}
        return BatchFeature(data=output_dict)

    @torch.no_grad()
    def get_action(
        self, backbone_output: BatchFeature, action_input: BatchFeature
    ) -> BatchFeature:
        backbone_output = self.process_backbone_output(backbone_output)

        vl_embs = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        state_features = self.state_encoder(action_input.state, embodiment_id)

        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.config.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        for t in range(num_steps):
            t_cont = t / float(num_steps)
            t_discretized = int(t_cont * self.num_timestep_buckets)

            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)

            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Build sequence: [state | actlat_tokens | action]
            parts = [state_features]
            if self.config.actlat_num_tokens > 0:
                al_tokens = self.actlat_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1)
                parts.append(al_tokens)
            parts.append(action_features)
            sa_embs = torch.cat(parts, dim=1)

            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=timesteps_tensor,
            )
            pred = self.action_decoder(model_output, embodiment_id)
            pred_velocity = pred[:, -self.action_horizon:]

            actions = actions + dt * pred_velocity

        return BatchFeature(data={"action_pred": actions})

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
