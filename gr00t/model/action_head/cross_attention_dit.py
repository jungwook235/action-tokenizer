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

from typing import Optional

import torch
import torch.nn.functional as F
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.embeddings import (
    SinusoidalPositionalEmbedding,
    TimestepEmbedding,
    Timesteps,
)
from torch import nn


class TimestepEncoder(nn.Module):
    def __init__(self, embedding_dim, compute_dtype=torch.float32):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timesteps):
        dtype = next(self.parameters()).dtype
        timesteps_proj = self.time_proj(timesteps).to(dtype)
        timesteps_emb = self.timestep_embedder(timesteps_proj)  # (N, D)
        return timesteps_emb


class AdaLayerNorm(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        chunk_dim: int = 0,
    ):
        super().__init__()
        self.chunk_dim = chunk_dim
        output_dim = embedding_dim * 2
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim // 2, norm_eps, norm_elementwise_affine)

    def forward(
        self,
        x: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        temb = self.linear(self.silu(temb))
        scale, shift = temb.chunk(2, dim=1)
        x = self.norm(x) * (1 + scale[:, None]) + shift[:, None]
        return x


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.dropout = dropout
        self.cross_attention_dim = cross_attention_dim
        self.activation_fn = activation_fn
        self.attention_bias = attention_bias
        self.norm_elementwise_affine = norm_elementwise_affine
        self.positional_embeddings = positional_embeddings
        self.num_positional_embeddings = num_positional_embeddings
        self.norm_type = norm_type

        if positional_embeddings and (num_positional_embeddings is None):
            raise ValueError(
                "If `positional_embedding` type is defined, `num_positition_embeddings` must also be defined."
            )

        if positional_embeddings == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding(
                dim, max_seq_length=num_positional_embeddings
            )
        else:
            self.pos_embed = None

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        if norm_type == "ada_norm":
            self.norm1 = AdaLayerNorm(dim)
        else:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
        )

        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )
        if final_dropout:
            self.final_dropout = nn.Dropout(dropout)
        else:
            self.final_dropout = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:

        # 0. Self-Attention
        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, temb)
        else:
            norm_hidden_states = self.norm1(hidden_states)

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            # encoder_attention_mask=encoder_attention_mask,
        )
        if self.final_dropout:
            attn_output = self.final_dropout(attn_output)

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 4. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        return hidden_states


class DiT(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        output_dim: int = 26,
        num_layers: int = 12,
        dropout: float = 0.1,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: Optional[int] = 1000,
        upcast_attention: bool = False,
        norm_type: str = "ada_norm",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        max_num_positional_embeddings: int = 512,
        compute_dtype=torch.float32,
        final_dropout: bool = True,
        positional_embeddings: Optional[str] = "sinusoidal",
        interleave_self_attention=False,
        cross_attention_dim: Optional[int] = None,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.gradient_checkpointing = False

        # Timestep encoder
        self.timestep_encoder = TimestepEncoder(
            embedding_dim=self.inner_dim, compute_dtype=self.config.compute_dtype
        )

        all_blocks = []
        for idx in range(self.config.num_layers):

            use_self_attn = idx % 2 == 1 and interleave_self_attention
            curr_cross_attention_dim = cross_attention_dim if not use_self_attn else None

            all_blocks += [
                BasicTransformerBlock(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    activation_fn=self.config.activation_fn,
                    attention_bias=self.config.attention_bias,
                    upcast_attention=self.config.upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                    positional_embeddings=positional_embeddings,
                    num_positional_embeddings=self.config.max_num_positional_embeddings,
                    final_dropout=final_dropout,
                    cross_attention_dim=curr_cross_attention_dim,
                )
            ]
        self.transformer_blocks = nn.ModuleList(all_blocks)

        # Output blocks
        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
        self.proj_out_2 = nn.Linear(self.inner_dim, self.config.output_dim)
        print(
            "Total number of DiT parameters: ",
            sum(p.numel() for p in self.parameters() if p.requires_grad),
        )
        
        # ============================================================================ #
        # Attention Map Analysis (분석용 - 내부 연산과 완전히 분리됨)
        # ============================================================================ #
        self._attention_maps = []  # 저장된 attention maps
        self._attention_hooks = []  # 등록된 hooks
        self._attention_logging_enabled = False  # attention logging 활성화 여부
        # ============================================================================ #

    def forward(
        self,
        hidden_states: torch.Tensor,  # Shape: (B, T, D)
        encoder_hidden_states: torch.Tensor,  # Shape: (B, S, D)
        timestep: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_all_hidden_states: bool = False,
    ):
        # Encode timesteps
        temb = self.timestep_encoder(timestep)

        # Process through transformer blocks - single pass through the blocks
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()

        all_hidden_states = [hidden_states]

        # Process through transformer blocks
        for idx, block in enumerate(self.transformer_blocks):
            if idx % 2 == 1 and self.config.interleave_self_attention:
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    temb=temb,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=None,
                    temb=temb,
                )
            all_hidden_states.append(hidden_states)

        # Output processing
        conditioning = temb
        shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        if return_all_hidden_states:
            return self.proj_out_2(hidden_states), all_hidden_states
        else:
            return self.proj_out_2(hidden_states)

    # ============================================================================ #
    # Attention Map Analysis Methods (분석용 - 내부 연산과 완전히 분리됨)
    # ============================================================================ #
    
    def _create_attention_hook(self, layer_idx: int):
        """
        특정 layer의 attention weights를 캡처하는 hook을 생성합니다.
        이 hook은 내부 연산을 전혀 변경하지 않고, 단순히 attention weights만 기록합니다.
        
        Attention weights는 Q·K^T로 계산되는, Value에 곱해지는 확률 분포입니다.
        예: 1번 토큰이 2~800번 토큰에 대한 attention 정도를 나타내는 map
        """
        def hook(module, input, output):
            # Custom processor가 module._last_attention_probs에 저장한 attention weights를 읽음
            if hasattr(module, '_last_attention_probs'):
                attn_weights = module._last_attention_probs
                
                # Attention weights 저장
                # Shape: (batch_size * num_heads, seq_len, seq_len)
                # 각 토큰(행)이 다른 모든 토큰(열)에 대한 attention 정도
                self._attention_maps.append({
                    'layer_idx': layer_idx,
                    'attention_weights': attn_weights.detach().cpu()  # CPU로 이동하여 메모리 절약
                })
        return hook
    
    def enable_attention_logging(self):
        """
        Attention map 로깅을 활성화합니다.
        각 transformer block의 attention 모듈에 hook을 등록하여 attention weights를 캡처합니다.
        
        주의: 이 메서드는 내부 연산을 변경하지 않으며, 단순히 분석을 위한 정보만 수집합니다.
        """
        if self._attention_logging_enabled:
            print("Attention logging is already enabled.")
            return
        
        self._attention_maps = []
        self._attention_hooks = []
        
        # 각 transformer block의 attention 모듈에 hook 등록
        # 중요: Self-attention 레이어만 로깅 (홀수 인덱스 1, 3, 5, 7, ...)
        # 짝수 인덱스는 cross-attention이므로 토큰 수가 달라 분석 불가
        for idx, block in enumerate(self.transformer_blocks):
            # Self-attention 레이어만 처리 (홀수 인덱스 또는 interleave_self_attention이 False일 때)
            is_self_attn_layer = (idx % 2 == 1 and self.config.interleave_self_attention) or not self.config.interleave_self_attention
            
            # BasicTransformerBlock의 attn1 (self-attention만 로깅)
            if hasattr(block, 'attn1') and is_self_attn_layer:
                # Attention processor를 커스텀 버전으로 교체
                # 기존 processor를 저장
                original_processor = block.attn1.processor
                
                # Custom processor로 교체하여 attention weights를 module 속성으로 저장
                from diffusers.models.attention_processor import AttnProcessor
                
                class AttnProcessorWithWeights(AttnProcessor):
                    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
                        # 원래 processor 로직 실행
                        batch_size, sequence_length, _ = hidden_states.shape
                        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                        
                        query = attn.to_q(hidden_states)
                        
                        if encoder_hidden_states is None:
                            encoder_hidden_states = hidden_states
                        
                        key = attn.to_k(encoder_hidden_states)
                        value = attn.to_v(encoder_hidden_states)
                        
                        query = attn.head_to_batch_dim(query)
                        key = attn.head_to_batch_dim(key)
                        value = attn.head_to_batch_dim(value)
                        
                        # Attention weights 계산 (Q·K^T로 만들어지는, Value에 곱해지는 그 가중치)
                        attention_probs = attn.get_attention_scores(query, key, attention_mask)
                        
                        # Attention weights를 module 속성으로 저장 (hook에서 읽을 수 있도록)
                        attn._last_attention_probs = attention_probs.detach()
                        
                        # Attention 적용
                        hidden_states = torch.bmm(attention_probs, value)
                        hidden_states = attn.batch_to_head_dim(hidden_states)
                        
                        # Linear projection and dropout
                        hidden_states = attn.to_out[0](hidden_states)
                        hidden_states = attn.to_out[1](hidden_states)
                        
                        # 원래대로 hidden_states만 반환 (tuple 반환 X, dropout 에러 방지)
                        return hidden_states
                
                # 새로운 processor 설정 (기존 processor는 저장)
                block.attn1._original_processor = original_processor
                block.attn1.processor = AttnProcessorWithWeights()
                
                # Hook 등록
                hook = self._create_attention_hook(idx)
                handle = block.attn1.register_forward_hook(hook)
                self._attention_hooks.append(handle)
        
        self._attention_logging_enabled = True
        print(f"Enabled attention logging for {len(self._attention_hooks)} layers.")
    
    def disable_attention_logging(self):
        """
        Attention map 로깅을 비활성화하고 모든 hooks를 제거합니다.
        원래의 attention processor로 복원합니다.
        """
        if not self._attention_logging_enabled:
            return
        
        # 모든 hooks 제거
        for handle in self._attention_hooks:
            handle.remove()
        self._attention_hooks = []
        
        # 원래 processor로 복원
        for block in self.transformer_blocks:
            if hasattr(block, 'attn1') and hasattr(block.attn1, '_original_processor'):
                block.attn1.processor = block.attn1._original_processor
                delattr(block.attn1, '_original_processor')
        
        self._attention_logging_enabled = False
        print("Disabled attention logging.")
    
    def get_attention_maps(self):
        """
        저장된 attention maps를 반환합니다.
        
        Returns:
            list: 각 layer의 attention maps를 포함하는 리스트
                  각 item은 {'layer_idx': int, 'attention_weights': Tensor} 형태
        """
        return self._attention_maps
    
    def clear_attention_maps(self):
        """
        저장된 attention maps를 모두 삭제합니다.
        """
        self._attention_maps = []
    
    def save_attention_maps(self, save_path: str):
        """
        저장된 attention maps를 파일로 저장합니다.
        
        Args:
            save_path: 저장할 파일 경로 (.pt, .npz, 또는 .png)
                      .png인 경우 attention maps를 시각화하여 저장합니다.
        """
        if not self._attention_maps:
            print("No attention maps to save.")
            return
        
        import os
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        if save_path.endswith('.pt'):
            torch.save(self._attention_maps, save_path)
            print(f"Saved {len(self._attention_maps)} attention maps to {save_path}")
        elif save_path.endswith('.npz'):
            import numpy as np
            save_dict = {}
            for i, attn_map in enumerate(self._attention_maps):
                if attn_map['attention_weights'] is not None:
                    save_dict[f"layer_{attn_map['layer_idx']}_step_{i}"] = attn_map['attention_weights'].numpy()
            np.savez_compressed(save_path, **save_dict)
            print(f"Saved {len(self._attention_maps)} attention maps to {save_path}")
        elif save_path.endswith('.png'):
            # PNG로 시각화하여 저장 (layer별로 개별 파일)
            import matplotlib.pyplot as plt
            import numpy as np
            
            num_layers = len(self._attention_maps)
            if num_layers == 0:
                print("No attention maps to visualize.")
                return
            
            # 파일명에서 확장자 분리
            base_path = save_path[:-4]  # .png 제거
            
            # 각 layer의 attention map을 개별 파일로 저장
            for idx, attn_map in enumerate(self._attention_maps):
                layer_idx = attn_map['layer_idx']
                attn_weights = attn_map['attention_weights']  # (B*H, L, L)
                
                # Average over batch and heads: (B*H, L, L) -> (L, L)
                # BFloat16 -> Float32 -> numpy (BFloat16은 numpy에서 지원하지 않음)
                avg_attn = attn_weights.mean(dim=0).float().numpy()
                
                # 개별 layer용 figure 생성
                fig, ax = plt.subplots(1, 1, figsize=(12, 10))
                
                # Heatmap 그리기
                im = ax.imshow(avg_attn, cmap='viridis', aspect='auto', interpolation='nearest')
                ax.set_title(f'Layer {layer_idx} - Attention Map', fontsize=14, fontweight='bold')
                ax.set_xlabel('Key Tokens', fontsize=12)
                ax.set_ylabel('Query Tokens', fontsize=12)
                
                # 토큰 번호 표시 (모든 토큰에 번호 표시)
                num_tokens = avg_attn.shape[0]
                
                # x축과 y축에 모든 토큰 번호 표시 (겹침 방지)
                if num_tokens <= 50:
                    # 토큰이 50개 이하면 모든 번호 표시
                    tick_positions = np.arange(num_tokens)
                    tick_labels = [str(i) for i in range(num_tokens)]
                    rotation = 0
                    fontsize = 8
                else:
                    # 토큰이 많으면 적절한 간격으로 표시 (겹침 방지를 위해 더 sparse하게)
                    step = max(1, num_tokens // 15)  # 30에서 15로 변경 (더 적은 레이블)
                    tick_positions = np.arange(0, num_tokens, step)
                    tick_labels = [str(i) for i in tick_positions]
                    rotation = 90  # 45도에서 90도로 변경 (완전 세로)
                    fontsize = 6  # 폰트 크기 감소
                
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(tick_labels, fontsize=fontsize, rotation=rotation, 
                                  ha='center' if rotation == 90 else ('right' if rotation > 0 else 'center'),
                                  va='top' if rotation == 90 else 'center')
                ax.set_yticks(tick_positions)
                ax.set_yticklabels(tick_labels, fontsize=fontsize)
                
                # 토큰 그룹 구분선 그리기 및 통계 계산
                # state_features(1) | future_tokens(64) | reasoning_tokens(64) | action_features(16)
                # 구분선 위치: 1, 65, 129
                attention_stats = {}  # Action query tokens의 attention 통계
                
                if num_tokens > 100:  # 전체 토큰이 충분히 많을 때만 구분선 그리기
                    separator_positions = [1, 65, 129]
                    for sep_pos in separator_positions:
                        if sep_pos < num_tokens:
                            # 세로선 (그룹 구분)
                            ax.axvline(x=sep_pos - 0.5, color='white', linewidth=2, linestyle='--', alpha=0.7)
                            # 가로선 (그룹 구분)
                            ax.axhline(y=sep_pos - 0.5, color='white', linewidth=2, linestyle='--', alpha=0.7)
                    
                    # 그룹 레이블 추가 (상단)
                    group_labels = [
                        ('State', 0, 1),
                        ('Future', 1, 65),
                        ('Reasoning', 65, 129),
                        ('Action', 129, num_tokens)
                    ]
                    for label, start, end in group_labels:
                        if end <= num_tokens:
                            mid_point = (start + end) / 2
                            ax.text(mid_point, -num_tokens * 0.03, label, 
                                   ha='center', va='bottom', fontsize=9, 
                                   fontweight='bold', color='darkblue')
                            # 좌측에도 레이블
                            ax.text(-num_tokens * 0.03, mid_point, label, 
                                   ha='right', va='center', fontsize=9, 
                                   fontweight='bold', color='darkblue', rotation=90)
                    
                    # Action 토큰 그룹(129-145)의 query들이 각 key 그룹에 대한 평균 attention 계산
                    action_query_start = 129
                    action_query_end = num_tokens
                    
                    if action_query_end > action_query_start:
                        # Action query들의 각 key 그룹에 대한 평균 attention
                        action_queries = avg_attn[action_query_start:action_query_end, :]  # (16, L)
                        
                        attention_stats[f'layer_{layer_idx}'] = {
                            'state_attention_mean': float(action_queries[:, 0:1].mean()),
                            'future_attention_mean': float(action_queries[:, 1:65].mean()),
                            'reasoning_attention_mean': float(action_queries[:, 65:129].mean()),
                            'action_attention_mean': float(action_queries[:, 129:action_query_end].mean()),
                        }
                
                # Colorbar 추가
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Attention Weight', fontsize=10)
                
                # 그리드 제거 (구분선이 더 명확함)
                ax.grid(False)
                
                plt.tight_layout()
                
                # Layer별 개별 파일로 저장
                layer_save_path = f"{base_path}_layer_{layer_idx}.png"
                plt.savefig(layer_save_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"Saved Layer {layer_idx} attention map to {layer_save_path}")
                
                # Attention 통계를 JSON으로 저장
                if attention_stats:
                    stats_save_path = f"{base_path}_layer_{layer_idx}_stats.json"
                    import json
                    with open(stats_save_path, 'w') as f:
                        json.dump(attention_stats, f, indent=2)
                    print(f"Saved Layer {layer_idx} attention statistics to {stats_save_path}")
            
            print(f"Saved {num_layers} attention map visualizations")
        else:
            raise ValueError("save_path must end with .pt, .npz, or .png")
        


    # ============================================================================ #


class SelfAttentionTransformer(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        output_dim: int = 26,
        num_layers: int = 12,
        dropout: float = 0.1,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: Optional[int] = 1000,
        upcast_attention: bool = False,
        max_num_positional_embeddings: int = 512,
        compute_dtype=torch.float32,
        final_dropout: bool = True,
        positional_embeddings: Optional[str] = "sinusoidal",
        interleave_self_attention=False,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.gradient_checkpointing = False

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    activation_fn=self.config.activation_fn,
                    attention_bias=self.config.attention_bias,
                    upcast_attention=self.config.upcast_attention,
                    positional_embeddings=positional_embeddings,
                    num_positional_embeddings=self.config.max_num_positional_embeddings,
                    final_dropout=final_dropout,
                )
                for _ in range(self.config.num_layers)
            ]
        )
        print(
            "Total number of SelfAttentionTransformer parameters: ",
            sum(p.numel() for p in self.parameters() if p.requires_grad),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,  # Shape: (B, T, D)
        return_all_hidden_states: bool = False,
    ):
        # Process through transformer blocks - single pass through the blocks
        hidden_states = hidden_states.contiguous()
        all_hidden_states = [hidden_states]

        # Process through transformer blocks
        for idx, block in enumerate(self.transformer_blocks):
            hidden_states = block(hidden_states)
            all_hidden_states.append(hidden_states)

        if return_all_hidden_states:
            return hidden_states, all_hidden_states
        else:
            return hidden_states




# class MMDiTBlock(Module):
#     def __init__(
#         self,
#         *,
#         dim_text,
#         dim_image,
#         dim_cond = None,
#         dim_head = 64,
#         heads = 8,
#         qk_rmsnorm = False,
#         flash_attn = False,
#         num_residual_streams = 1,
#         ff_kwargs: dict = dict()
#     ):
#         super().__init__()

#         # residual functions / maybe hyper connections

#         residual_klass = Residual if num_residual_streams == 1 else HyperConnections

#         self.text_attn_residual_fn = residual_klass(num_residual_streams, dim = dim_text)
#         self.text_ff_residual_fn = residual_klass(num_residual_streams, dim = dim_text)

#         self.image_attn_residual_fn = residual_klass(num_residual_streams, dim = dim_image)
#         self.image_ff_residual_fn = residual_klass(num_residual_streams, dim = dim_image)

#         # handle optional time conditioning

#         has_cond = exists(dim_cond)
#         self.has_cond = has_cond

#         if has_cond:
#             dim_gammas = (
#                 *((dim_text,) * 4),
#                 *((dim_image,) * 4)
#             )

#             dim_betas = (
#                 *((dim_text,) * 2),
#                 *((dim_image,) * 2),
#             )

#             self.cond_dims = (*dim_gammas, *dim_betas)

#             to_cond_linear = nn.Linear(dim_cond, sum(self.cond_dims))

#             self.to_cond = nn.Sequential(
#                 Rearrange('b d -> b 1 d'),
#                 nn.SiLU(),
#                 to_cond_linear
#             )

#             nn.init.zeros_(to_cond_linear.weight)
#             nn.init.zeros_(to_cond_linear.bias)
#             nn.init.constant_(to_cond_linear.bias[:sum(dim_gammas)], 1.)

#         # handle adaptive norms

#         self.text_attn_layernorm = nn.LayerNorm(dim_text, elementwise_affine = not has_cond)
#         self.image_attn_layernorm = nn.LayerNorm(dim_image, elementwise_affine = not has_cond)

#         self.text_ff_layernorm = nn.LayerNorm(dim_text, elementwise_affine = not has_cond)
#         self.image_ff_layernorm = nn.LayerNorm(dim_image, elementwise_affine = not has_cond)

#         # attention and feedforward

#         self.joint_attn = JointAttention(
#             dim_inputs = (dim_text, dim_image),
#             dim_head = dim_head,
#             heads = heads,
#             flash = flash_attn
#         )

#         self.text_ff = FeedForward(dim_text, **ff_kwargs)
#         self.image_ff = FeedForward(dim_image, **ff_kwargs)

#     def forward(
#         self,
#         *,
#         text_tokens,
#         image_tokens,
#         text_mask = None,
#         time_cond = None,
#         skip_feedforward_text_tokens = True
#     ):
#         assert not (exists(time_cond) ^ self.has_cond), 'time condition must be passed in if dim_cond is set at init. it should not be passed in if not set'

#         if self.has_cond:
#             (
#                 text_pre_attn_gamma,
#                 text_post_attn_gamma,
#                 text_pre_ff_gamma,
#                 text_post_ff_gamma,
#                 image_pre_attn_gamma,
#                 image_post_attn_gamma,
#                 image_pre_ff_gamma,
#                 image_post_ff_gamma,
#                 text_pre_attn_beta,
#                 text_pre_ff_beta,
#                 image_pre_attn_beta,
#                 image_pre_ff_beta,
#             ) = self.to_cond(time_cond).split(self.cond_dims, dim = -1)

#         # handle attn adaptive layernorm

#         text_tokens, add_text_residual = self.text_attn_residual_fn(text_tokens)
#         image_tokens, add_image_residual = self.image_attn_residual_fn(image_tokens)

#         text_tokens = self.text_attn_layernorm(text_tokens)
#         image_tokens = self.image_attn_layernorm(image_tokens)

#         if self.has_cond:
#             text_tokens = text_tokens * text_pre_attn_gamma + text_pre_attn_beta
#             image_tokens = image_tokens * image_pre_attn_gamma + image_pre_attn_beta

#         # attention

#         text_tokens, image_tokens = self.joint_attn(
#             inputs = (text_tokens, image_tokens),
#             masks = (text_mask, None)
#         )

#         # condition attention output

#         if self.has_cond:
#             text_tokens = text_tokens * text_post_attn_gamma
#             image_tokens = image_tokens * image_post_attn_gamma

#         # add attention residual

#         text_tokens = add_text_residual(text_tokens)
#         image_tokens = add_image_residual(image_tokens)

#         # handle feedforward adaptive layernorm

#         if not skip_feedforward_text_tokens:
#             text_tokens, add_text_residual = self.text_ff_residual_fn(text_tokens)
#             text_tokens = self.text_ff_layernorm(text_tokens)

#             if self.has_cond:
#                 text_tokens = text_tokens * text_pre_ff_gamma + text_pre_ff_beta

#         image_tokens, add_image_residual = self.image_ff_residual_fn(image_tokens)
#         image_tokens = self.image_ff_layernorm(image_tokens)

#         if self.has_cond:
#             image_tokens = image_tokens * image_pre_ff_gamma + image_pre_ff_beta

#         # images feedforward

#         image_tokens = self.image_ff(image_tokens)

#         # images condition feedforward output

#         if self.has_cond:
#             image_tokens = image_tokens * image_post_ff_gamma

#         # images feedforward residual

#         image_tokens = add_image_residual(image_tokens)

#         # early return, for last block in mmdit

#         if skip_feedforward_text_tokens:
#             return text_tokens, image_tokens

#         # text feedforward

#         text_tokens = self.text_ff(text_tokens)

#         # text condition feedforward output

#         if self.has_cond:
#             text_tokens = text_tokens * text_post_ff_gamma

#         # text feedforward residual

#         text_tokens = add_text_residual(text_tokens)

#         # return

#         return text_tokens, image_tokens
    
