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
    num_target_vision_tokens: int = field(
        default=32, metadata={"help": "Number of target vision tokens."}
    )
    flare_loss_lambda: float = field(default=1.0, metadata={"help": "Lambda for the flare loss."})
    flare_align_layers: int = field(default=12, metadata={"help": "Layer number from DiT layers to align (Max 16)."})

    image_count: int = field(default=1, metadata={"help": "Number of images to use for training."})
    video_only: bool = field(default=False, metadata={"help": "Whether to only use video for training."})
    flare_image_time: str = field(default="future", metadata={"help": "Whether to use future or current image for FLARE. Options: 'future', 'current'."})

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
        print("num_target_vision_tokens: ", config.num_target_vision_tokens)
        if config.num_target_vision_tokens > 0:
            print("Using future vision tokens in FLARE loss. num_target_vision_tokens =", config.num_target_vision_tokens)
            self.future_tokens = nn.Embedding(config.num_target_vision_tokens, self.input_embedding_dim)
            nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

            self.siglip_embedding_size = 1152
            self.flare_projector = nn.Sequential(
                nn.Linear(self.input_embedding_dim, self.siglip_embedding_size),
                nn.SiLU(),
                nn.Linear(self.siglip_embedding_size, self.siglip_embedding_size),
                nn.SiLU(),
                nn.Linear(self.siglip_embedding_size, self.siglip_embedding_size)
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
        print(f"Tune action head projector: {self.tune_projector}")
        print(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_projector and not tune_diffusion_model:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No action head trainable parameters found.")
    
    def _init_weights(self, module, name):
        if "weight" in name:
            # torch.nn.init.normal_(module, mean=0.0, std=0.02)
            init.kaiming_uniform_(module, a=math.sqrt(5))
        if "bias" in name:
            weight_name = name.replace("bias", "weight")
            weight = self.get_parameter(weight_name)

            fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(module, -bound, bound)

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
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

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature, future_backbone_output: Optional[BatchFeature]) -> BatchFeature:
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        backbone_output = self.process_backbone_output(backbone_output)

        if self.config.expand_batch is not None:
            for k, v in backbone_output.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                backbone_output[k] = expanded

            for k, v in action_input.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                action_input[k] = expanded

        # Get vision and language embeddings.
        vl_embs = backbone_output.backbone_features
        device = vl_embs.device

        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Embed noised action trajectory.
        actions = action_input.action
        B = actions.shape[0]
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        if self.config.video_only:
            noisy_trajectory = noise
            
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Join vision, language, state and action embedding along sequence dimension.
        if self.config.num_target_vision_tokens > 0:
            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
            sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)
        else:
            sa_embs = torch.cat((state_features, action_features), dim=1)

        vl_attn_mask = backbone_output.backbone_attention_mask

        model_output, model_hidden_states = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=vl_attn_mask,
            timestep=t_discretized,
            return_all_hidden_states=True,  # NOTE (YL): not using flare now
        )
        flare_hidden_states = model_hidden_states[self.config.flare_align_layers] # [16, 49, 1536]

        if self.config.num_target_vision_tokens > 0:
            flare_hidden_states = flare_hidden_states[:, -actions.shape[1]-self.config.num_target_vision_tokens:-actions.shape[1],:]
            

            align_flare_hidden_states = self.flare_projector(flare_hidden_states)


        # print("flare_hidden_states has nan", torch.isnan(flare_hidden_states).any())
        # print("align_flare_hidden_states has nan", torch.isnan(align_flare_hidden_states).any())

            if self.config.flare_image_time == "current":
                source_vit_embeds = backbone_output.raw_vit_embeds
            else:
                source_vit_embeds = future_backbone_output.raw_vit_embeds

            if source_vit_embeds.shape[0] == B * 3:
                vision_target = source_vit_embeds[0::3] # [16, 256, 1152]
                vision_target_2 = source_vit_embeds[1::3] # [16, 256, 1152]
                vision_target_3 = source_vit_embeds[2::3] # [16, 256, 1152]
            elif source_vit_embeds.shape[0] == B * 2:
                vision_target = source_vit_embeds[0::2] # [16, 256, 1152]
                vision_target_2 = source_vit_embeds[1::2] # [16, 256, 1152]
                vision_target_3 = None
            elif source_vit_embeds.shape[0] == B:
                vision_target = source_vit_embeds
                vision_target_2 = None
                vision_target_3 = None
        
        # if future_backbone_output.raw_vit_embeds.shape[0] % 3 == 0:
        #     align_target = future_backbone_output.raw_vit_embeds[0::3]
        # else:
        #     align_target = future_backbone_output.raw_vit_embeds[0::2]
        align_target = None
        if self.config.num_target_vision_tokens == 64 * self.config.image_count:
            patch_h = int((vision_target.shape[1]) ** 0.5)
            vision_target = vision_target.view(vision_target.shape[0], patch_h, patch_h, -1)
            vision_target = vision_target.permute(0, 3, 1, 2)
            vision_target = F.avg_pool2d(vision_target, kernel_size=2, stride=2)
            vision_target = vision_target.permute(0, 2, 3, 1).view(vision_target.shape[0], 64, -1)
            
            if self.config.image_count >= 2:
                vision_target_2 = vision_target_2.view(vision_target_2.shape[0], patch_h, patch_h, -1)
                vision_target_2 = vision_target_2.permute(0, 3, 1, 2)
                vision_target_2 = F.avg_pool2d(vision_target_2, kernel_size=2, stride=2)
                vision_target_2 = vision_target_2.permute(0, 2, 3, 1).view(vision_target_2.shape[0], 64, -1)
                
            if self.config.image_count >= 3:
                vision_target_3 = vision_target_3.view(vision_target_3.shape[0], patch_h, patch_h, -1)
                vision_target_3 = vision_target_3.permute(0, 3, 1, 2)
                vision_target_3 = F.avg_pool2d(vision_target_3, kernel_size=2, stride=2)
                vision_target_3 = vision_target_3.permute(0, 2, 3, 1).view(vision_target_3.shape[0], 64, -1)
                
            if self.config.image_count == 3:
                align_target = torch.cat((vision_target, vision_target_2, vision_target_3), dim=1)
            elif self.config.image_count == 2:
                align_target = torch.cat((vision_target, vision_target_2), dim=1)
            elif self.config.image_count == 1:
                align_target = vision_target
                
                
        #patch_h = int((align_target.shape[1] // self.config.image_count) ** 0.5)

        # if self.config.num_target_vision_tokens == 64:
        #     patch_h = int(align_target.shape[1] ** 0.5)
        #     align_target = align_target.view(align_target.shape[0], patch_h, patch_h, -1)
        #     align_target = align_target.permute(0, 3, 1, 2)
        #     align_target = F.avg_pool2d(align_target, kernel_size=2, stride=2)
        #     align_target = align_target.permute(0, 2, 3, 1).view(align_target.shape[0], 64, -1)

        # print("align_flare_hidden_states", align_flare_hidden_states.shape)
        # print("align_target", align_target.shape)

        # future_tokens torch.Size([16, 32, 1536])
        # backbone_features torch.Size([16, 560, 2048])
        # backbone_attention_mask torch.Size([16, 560])
        # 17

        # print(align_flare_hidden_states.shape, align_target.shape)
        # exit()

        # print("align_flare_hidden_states", align_flare_hidden_states.shape)
        # print("align_target", align_target.shape)

        if align_target is not None and (align_flare_hidden_states.shape == align_target.shape):
            flare_loss = -F.cosine_similarity(align_flare_hidden_states, align_target)
            flare_loss = flare_loss.mean()
            flare_loss = flare_loss * self.config.flare_loss_lambda
        else:
            #flare_loss = align_flare_hidden_states.mean() + align_target.mean()
            flare_loss = 0.0

        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1] :]

        # Slice out only the action portion of pred and target.
        action_mask = action_input.action_mask
        loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        loss = loss.sum() / action_mask.sum()

        if self.config.video_only:
            loss *= 0.0

        # print("pred_actions has nan", torch.isnan(pred_actions).any())
        # print("velocity has nan", torch.isnan(velocity).any())

        # print("align_flare_hidden_states has nan", torch.isnan(align_flare_hidden_states).any())
        # print("align_target has nan", torch.isnan(align_target).any())

        #print("flare_loss", flare_loss, flush=True)
        #print("loss", loss, flush=True)
        # exit()
        # print("align_flare_hidden_states", align_flare_hidden_states.shape)
        # print("align_target", align_target.shape)
        # exit()
        
        loss = loss + flare_loss
        output_dict = {
            "loss": loss,
        }
        #print("action head loss:", loss, flush=True)
        return BatchFeature(data=output_dict)

    @torch.no_grad()
    def get_action(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:

        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embs = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Set initial actions as the sampled noise.
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.config.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        # Run denoising steps.
        for t in range(num_steps):
            t_cont = t / float(num_steps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            # Maybe add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Join vision, language, state and action embedding along sequence dimension.
            if self.config.num_target_vision_tokens > 0:
                future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
                sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)
            else:
                sa_embs = torch.cat((state_features, action_features), dim=1)
            # Run model forward.
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=timesteps_tensor,
            )
            pred = self.action_decoder(model_output, embodiment_id)

            pred_velocity = pred[:, -self.action_horizon :]

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity
        return BatchFeature(data={"action_pred": actions})

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
