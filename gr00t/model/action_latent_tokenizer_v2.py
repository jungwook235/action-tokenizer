"""
Action Latent Tokenizer V2.

v1과 다른 점:
- Recon: 하나의 decoder에 global, time, hand token을 모두 concat하여 입력.
         출력은 action만 reconstruct.
- Hand State Prediction: cross-attention decoder로 hand token을 condition으로
         현재 hand state → 미래 hand state를 예측.
- Masked Latent Recon: encoder의 action latent 일부를 마스킹하고
         동일 recon decoder로 전체 action을 reconstruct.
- Global Token Learning: FAST tokenizer로 action을 discrete token화 →
         VTP-style text encoder → contrastive 또는 regression loss.

지원하는 loss:
  1. Recon (time + global + hand tokens → action reconstruct)
  2. Hand state prediction (hand tokens → future hand states)
  3. Masked latent recon (masked encoder output → full action reconstruct)
  4. Global token contrastive / regression
  5. Frequency domain loss (optional)

Building blocks는 v1과 동일 (cross-repo import 회피).
"""

import numpy as np
from functools import partial
from typing import Optional

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# Building blocks (oat/oat/action_latent/v1/model/)
# =====================================================================


class Fp32LayerNorm(nn.LayerNorm):
    def forward(self, input):
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0, bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))


class GatedMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.0, bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = int(2 * (hidden_features or in_features) / 3)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.fc3 = nn.Linear(in_features, hidden_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x)) * self.fc3(x)))


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, proj_bias=False, proj_drop=0.0,
                 qk_norm=True, norm_layer=partial(Fp32LayerNorm, bias=False, elementwise_affine=False)):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        if qk_norm:
            self.q_norm = norm_layer(self.head_dim)
            self.k_norm = norm_layer(self.head_dim)
        else:
            self.q_norm = self.k_norm = nn.Identity()

    def forward(self, x, block_mask=None, **kwargs):
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        q = einops.rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = einops.rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = einops.rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)
        q, k = self.q_norm(q), self.k_norm(k)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=block_mask)
        x = einops.rearrange(x, "b h n d -> b n (h d)")
        return self.proj_drop(self.proj(x))


class Block(nn.Module):
    def __init__(self, dim, num_heads=None, head_dim=None, mlp_ratio=4.0,
                 qkv_bias=False, proj_bias=False, mlp_bias=False, drop=0.0, drop_path=0.0,
                 act_layer=nn.GELU, norm_layer=partial(Fp32LayerNorm, bias=False, elementwise_affine=False),
                 gated_mlp=False, qk_norm=False, **kwargs):
        super().__init__()
        self.norm1 = norm_layer(dim)
        num_heads = num_heads or dim // head_dim
        self.attn = SelfAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias,
                                  proj_drop=drop, qk_norm=qk_norm, norm_layer=norm_layer)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_layer = GatedMlp if gated_mlp else Mlp
        self.mlp = mlp_layer(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, bias=mlp_bias, drop=drop)

    def forward(self, x, block_mask=None, **kwargs):
        x = x + self.drop_path(self.attn(self.norm1(x), block_mask=block_mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, dim=768, depth=12, head_dim=64, mlp_ratio=4.0, qkv_bias=False,
                 proj_bias=False, mlp_bias=False, drop=0.0, drop_path_rate=0.0,
                 act_layer=nn.SiLU, norm_layer=partial(Fp32LayerNorm, bias=False, elementwise_affine=False),
                 gated_mlp=True, qk_norm=True, weight_init_style="xavier"):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=dim, head_dim=head_dim, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  proj_bias=proj_bias, mlp_bias=mlp_bias, drop=drop, drop_path=dpr[i],
                  act_layer=act_layer, norm_layer=norm_layer, gated_mlp=gated_mlp, qk_norm=qk_norm)
            for i in range(depth)
        ])
        self.weight_init_style = weight_init_style
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.weight_init_style == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                elif self.weight_init_style == "trunc_normal":
                    nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, block_mask=None):
        for block in self.blocks:
            x = block(x, block_mask=block_mask)
        return x


def build_1d_sincos_posemb(L, embed_dim=1024, temperature=10000.0):
    assert embed_dim % 2 == 0
    pos = torch.arange(L, dtype=torch.float32)
    dim_t = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega = 1.0 / (temperature ** (dim_t / (embed_dim // 2)))
    out = torch.einsum("n,d->nd", pos, omega)
    pos_emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1)
    return pos_emb.transpose(0, 1).unsqueeze(0).contiguous()


class PositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_sizes: list[int]):
        super().__init__()
        assert len(max_sizes) == 1
        self.posembs = nn.Parameter(build_1d_sincos_posemb(max_sizes[0], embed_dim=dim), requires_grad=False)

    def forward(self, shape: list[int]) -> torch.Tensor:
        return self.posembs[:, :, :shape[0]]


class PositionalEmbeddingAdder(nn.Module):
    def __init__(self, dim: int, max_sizes: list[int]):
        super().__init__()
        assert len(max_sizes) == 1
        self.posembs = nn.Parameter(build_1d_sincos_posemb(max_sizes[0], embed_dim=dim), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.shape[1]
        pe = einops.rearrange(self.posembs[:, :, :L], "1 D L -> 1 L D")
        return x + pe


class LinearHead(nn.Module):
    def __init__(self, dim: int, dim_out: int, weight_init_style: str = "zero"):
        super().__init__()
        self.norm = Fp32LayerNorm(dim, bias=False, elementwise_affine=False)
        self.proj = nn.Linear(dim, dim_out, bias=True)
        if weight_init_style == "zero":
            nn.init.constant_(self.proj.weight, 0)
        elif weight_init_style == "xavier":
            nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(x))


class LinearLayer(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, weight_init_style: str = "xavier"):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=True)
        if weight_init_style == "xavier":
            nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# =====================================================================
# Encoder (with global/hand token support)
# =====================================================================


class TimeWiseEncoder(nn.Module):
    """Action chunk → per-timestep latent tokens (+ optional global/hand tokens).

    [B, T, D] → Linear(D, E) → PE(T) → [global | time | hand] → Transformer
    Output: global[B,Ng,E], time[B,T,E], hand[B,Nh,E]
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
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.emb_dim = emb_dim
        self.num_global_tokens = num_global_tokens
        self.num_hand_tokens = num_hand_tokens

        self.action_proj = nn.Linear(action_dim, emb_dim)
        self.time_pos_emb = PositionalEmbeddingAdder(emb_dim, max_sizes=[action_horizon])

        self.global_tokens = nn.Parameter(torch.randn(num_global_tokens, emb_dim)) if num_global_tokens > 0 else None
        self.hand_tokens = nn.Parameter(torch.randn(num_hand_tokens, emb_dim)) if num_hand_tokens > 0 else None

        self.transformer = Transformer(dim=emb_dim, depth=depth, head_dim=head_dim, drop=pdropout)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, T, D] normalized actions
        Returns:
            global_tokens: [B, Ng, E]
            time_tokens:   [B, T, E]
            hand_tokens:   [B, Nh, E]
        """
        B, T, D = x.shape
        Ng = self.num_global_tokens
        Nh = self.num_hand_tokens

        x = self.action_proj(x)       # [B, T, E]
        x = self.time_pos_emb(x)      # [B, T, E]

        parts = []
        if Ng > 0:
            parts.append(self.global_tokens.unsqueeze(0).expand(B, -1, -1))
        parts.append(x)
        if Nh > 0:
            parts.append(self.hand_tokens.unsqueeze(0).expand(B, -1, -1))
        x = torch.cat(parts, dim=1)  # [B, Ng+T+Nh, E]

        x = self.transformer(x)  # [B, Ng+T+Nh, E]

        global_out = x[:, :Ng]           # [B, Ng, E]
        time_out = x[:, Ng:Ng + T]       # [B, T, E]
        hand_out = x[:, Ng + T:]         # [B, Nh, E]

        return global_out, time_out, hand_out


# =====================================================================
# Decoder: Reconstruction (accepts global + time + hand tokens)
# =====================================================================


class ReconDecoder(nn.Module):
    """Reconstruction decoder.

    Input: time_tokens [B, T, E], optional global_tokens [B, Ng, E], optional hand_tokens [B, Nh, E]
    모든 토큰을 concat하여 입력하고, action space [B, T, D]를 reconstruct.
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
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.emb_dim = emb_dim
        self.decoder_mode = decoder_mode
        self.num_global_tokens = num_global_tokens
        self.num_hand_tokens = num_hand_tokens

        self.time_proj = LinearLayer(emb_dim, emb_dim)
        self.time_pos_emb = PositionalEmbeddingAdder(emb_dim, max_sizes=[action_horizon])
        self.head = LinearHead(emb_dim, action_dim)

        if num_global_tokens > 0:
            self.global_proj = LinearLayer(emb_dim, emb_dim)
            self.global_pos_emb = PositionalEmbeddingAdder(emb_dim, max_sizes=[num_global_tokens])
        else:
            self.global_proj = None
            self.global_pos_emb = None

        if num_hand_tokens > 0:
            self.hand_proj = LinearLayer(emb_dim, emb_dim)
            self.hand_pos_emb = PositionalEmbeddingAdder(emb_dim, max_sizes=[num_hand_tokens])
        else:
            self.hand_proj = None
            self.hand_pos_emb = None

        if decoder_mode == "cross_attention":
            self.query_pos_emb = PositionalEmbedding(emb_dim, max_sizes=[action_horizon])
            self.decoder = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(
                    d_model=emb_dim, nhead=emb_dim // head_dim,
                    dim_feedforward=4 * emb_dim, dropout=pdropout,
                    activation="gelu", batch_first=True, norm_first=True,
                ),
                num_layers=depth,
            )
        elif decoder_mode == "self_attention":
            self.transformer = Transformer(dim=emb_dim, depth=depth, head_dim=head_dim, drop=pdropout)
        else:
            raise ValueError(f"Unknown decoder_mode: {decoder_mode}")

    def forward(
        self,
        time_tokens: torch.Tensor,
        global_tokens: Optional[torch.Tensor] = None,
        hand_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, E = time_tokens.shape

        # Build memory/input by concatenating all available tokens
        parts = []

        if global_tokens is not None and self.global_proj is not None:
            g = self.global_pos_emb(self.global_proj(global_tokens))
            parts.append(g)

        latents = self.time_pos_emb(self.time_proj(time_tokens))
        parts.append(latents)

        if hand_tokens is not None and self.hand_proj is not None:
            h = self.hand_pos_emb(self.hand_proj(hand_tokens))
            parts.append(h)

        memory = torch.cat(parts, dim=1)  # [B, Ng+T+Nh, E]

        Ng = global_tokens.shape[1] if global_tokens is not None and self.global_proj is not None else 0

        if self.decoder_mode == "cross_attention":
            query = self.query_pos_emb(shape=[T])
            query = einops.rearrange(query, "1 E T -> 1 T E").expand(B, -1, -1)
            out = self.decoder(query, memory)
        else:
            out = self.transformer(memory)
            out = out[:, Ng:Ng + T]  # extract time positions only

        return self.head(out)  # [B, T, D]


# =====================================================================
# Hand State Prediction Decoder
# =====================================================================


class HandStatePredDecoder(nn.Module):
    """Cross-attention decoder for future hand state prediction.

    Query: 현재 hand state (복제하여 각 미래 step에 대해 하나씩, future pos emb 추가)
    KV: encoder의 hand tokens 또는 time tokens (num_kv_tokens로 일반화)
    Output: [B, num_future_steps, hand_state_dim] 미래 hand state 예측
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
    ):
        super().__init__()
        self.hand_state_dim = hand_state_dim
        self.emb_dim = emb_dim
        self.num_future_steps = num_future_steps

        self.state_proj = nn.Linear(hand_state_dim, emb_dim)
        self.future_pos_emb = nn.Parameter(torch.randn(1, num_future_steps, emb_dim) * 0.02)
        self.hand_kv_proj = LinearLayer(emb_dim, emb_dim)
        self.hand_kv_pos_emb = PositionalEmbeddingAdder(emb_dim, max_sizes=[num_kv_tokens])

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=emb_dim, nhead=emb_dim // head_dim,
                dim_feedforward=4 * emb_dim, dropout=pdropout,
                activation="gelu", batch_first=True, norm_first=True,
            ),
            num_layers=depth,
        )
        self.head = LinearHead(emb_dim, hand_state_dim)

    def forward(self, hand_state: torch.Tensor, kv_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hand_state: [B, hand_state_dim] 현재 hand state
            kv_tokens: [B, N, E] KV로 사용할 tokens (hand_tok 또는 time_tok)
        Returns:
            [B, num_future_steps, hand_state_dim] 미래 hand state 예측
        """
        B = hand_state.shape[0]

        # Query: 현재 state를 복제하여 각 future step에 하나씩
        q = self.state_proj(hand_state).unsqueeze(1).expand(-1, self.num_future_steps, -1)  # [B, F, E]
        q = q + self.future_pos_emb  # [B, F, E] — future position 구분

        # KV
        kv = self.hand_kv_pos_emb(self.hand_kv_proj(kv_tokens))  # [B, N, E]

        out = self.decoder(q, kv)  # [B, F, E]
        return self.head(out)  # [B, F, hand_state_dim]


# =====================================================================
# Action Text Encoder (VTP TextTransformer style, self-contained)
# =====================================================================


class ActionTextEncoder(nn.Module):
    """FAST tokenizer의 discrete action tokens → text feature.

    VTP의 TextTransformer 구조를 자체 포함:
    token_embedding → positional_embedding → Transformer (causal) → ln_final → pooling → projection
    """

    def __init__(
        self,
        vocab_size: int = 2048,
        context_length: int = 256,
        width: int = 256,
        heads: int = 4,
        layers: int = 4,
        output_dim: int = 256,
        pad_token_id: Optional[int] = None,
    ):
        super().__init__()
        self.context_length = context_length
        self.width = width
        self.output_dim = output_dim
        # pad_token_id defaults to vocab_size (one beyond valid range)
        self.pad_token_id = pad_token_id if pad_token_id is not None else vocab_size

        self.token_embedding = nn.Embedding(vocab_size + 1, width)  # +1 for pad token
        self.positional_embedding = nn.Parameter(torch.empty(context_length, width))

        self.transformer = Transformer(
            dim=width, depth=layers, head_dim=max(1, width // heads),
            drop=0.0, gated_mlp=True, qk_norm=True,
        )
        self.ln_final = Fp32LayerNorm(width, bias=False, elementwise_affine=False)

        # Matrix projection (VTP style)
        self.text_projection = nn.Parameter(torch.empty(width, output_dim))

        # Causal attention mask
        mask = torch.empty(context_length, context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        self.register_buffer("attn_mask", mask, persistent=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.normal_(self.text_projection, std=self.width ** -0.5)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [B, L] discrete token IDs (padded with pad_token_id)
        Returns:
            [B, output_dim] pooled text features (L2 normalized)
        """
        B, L = tokens.shape
        L = min(L, self.context_length)
        tokens = tokens[:, :L]

        x = self.token_embedding(tokens)  # [B, L, W]
        x = x + self.positional_embedding[:L]

        # Causal attention
        attn_mask = self.attn_mask[:L, :L]
        for block in self.transformer.blocks:
            x = block(x, block_mask=attn_mask)

        x = self.ln_final(x)

        # Pool: last non-pad token position
        # pad_token_id 위치가 아닌 마지막 유효 토큰
        non_pad = (tokens != self.pad_token_id)  # [B, L]
        # 각 batch에서 마지막 non-pad index 찾기
        # non_pad가 전부 False인 경우 방지
        seq_lens = non_pad.sum(dim=1).clamp(min=1)  # [B]
        pool_idx = seq_lens - 1  # [B]
        pooled = x[torch.arange(B, device=x.device), pool_idx]  # [B, W]

        # Project
        features = pooled @ self.text_projection  # [B, output_dim]
        return F.normalize(features, dim=-1)

    @classmethod
    def from_pretrained(cls, pretrained_path: str, **kwargs):
        """VTP pretrained weight 로드. key mapping 수행."""
        model = cls(**kwargs)
        state_dict = torch.load(pretrained_path, map_location="cpu", weights_only=False)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        # VTP key mapping: text. prefix를 제거
        mapped = {}
        for k, v in state_dict.items():
            # VTP stores as "text.token_embedding.weight" etc.
            if k.startswith("text."):
                new_k = k[len("text."):]
                # VTP Transformer uses "transformer.resblocks" → our Transformer uses "transformer.blocks"
                new_k = new_k.replace("resblocks", "blocks")
                mapped[new_k] = v

        # Load with strict=False (some keys may not match due to architecture differences)
        missing, unexpected = model.load_state_dict(mapped, strict=False)
        print(f"[ActionTextEncoder] Loaded pretrained. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        return model


# =====================================================================
# Global Token Loss Module
# =====================================================================


class GlobalTokenLossModule(nn.Module):
    """Global token과 text feature 간 contrastive 또는 regression loss.

    Contrastive: CLIP-style symmetric cross-entropy
    Regression: global tokens → MLP → MSE with text features
    """

    def __init__(
        self,
        mode: str = "contrastive",
        emb_dim: int = 256,
        text_feat_dim: int = 256,
        pool_type: str = "mean",
        num_global_tokens: int = 1,
    ):
        """
        Args:
            pool_type: global token pooling 방식
                - "mean": mean pooling
                - "max": max pooling
                - "attn": learnable attention pooling (weighted sum)
                - "linear": linear projection from Ng*E → E
        """
        super().__init__()
        self.mode = mode
        self.pool_type = pool_type

        # Attention pooling
        if pool_type == "attn":
            self.attn_pool = nn.Sequential(
                nn.Linear(emb_dim, 1),
            )
        elif pool_type == "linear" and num_global_tokens > 1:
            self.linear_pool = nn.Linear(num_global_tokens * emb_dim, emb_dim)
        else:
            self.attn_pool = None
            self.linear_pool = None

        if mode == "contrastive":
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            self.pool_proj = nn.Linear(emb_dim, text_feat_dim) if emb_dim != text_feat_dim else nn.Identity()
        elif mode == "regression":
            self.proj = nn.Sequential(
                nn.Linear(emb_dim, emb_dim),
                nn.SiLU(),
                nn.Linear(emb_dim, text_feat_dim),
            )
        else:
            raise ValueError(f"Unknown global_loss_mode: {mode}")

    def _pool_global(self, global_tokens: torch.Tensor) -> torch.Tensor:
        """[B, Ng, E] → [B, E]"""
        Ng = global_tokens.shape[1]
        if Ng == 1:
            return global_tokens.squeeze(1)

        if self.pool_type == "mean":
            return global_tokens.mean(dim=1)
        elif self.pool_type == "max":
            return global_tokens.max(dim=1).values
        elif self.pool_type == "attn":
            # Learnable attention weights
            weights = self.attn_pool(global_tokens)  # [B, Ng, 1]
            weights = F.softmax(weights, dim=1)
            return (global_tokens * weights).sum(dim=1)  # [B, E]
        elif self.pool_type == "linear":
            B, Ng, E = global_tokens.shape
            return self.linear_pool(global_tokens.reshape(B, Ng * E))  # [B, E]
        else:
            raise ValueError(f"Unknown pool_type: {self.pool_type}")

    def forward(self, global_tokens: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            global_tokens: [B, Ng, E]
            text_features: [B, text_feat_dim] (L2 normalized from ActionTextEncoder)
        Returns:
            scalar loss
        """
        pooled = self._pool_global(global_tokens)  # [B, E]

        if self.mode == "contrastive":
            pooled = self.pool_proj(pooled)
            pooled = F.normalize(pooled, dim=-1)
            # text_features already normalized

            logit_scale = self.logit_scale.exp()
            logits = logit_scale * pooled @ text_features.T  # [B, B]

            labels = torch.arange(logits.shape[0], device=logits.device)
            loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
            return loss

        else:  # regression
            pred = self.proj(pooled)  # [B, text_feat_dim]
            return F.mse_loss(pred, text_features.detach())


# =====================================================================
# Frequency domain loss
# =====================================================================


def frequency_loss(pred: torch.Tensor, target: torch.Tensor, loss_type: str = "mse") -> torch.Tensor:
    pred_fft = torch.fft.rfft(pred, dim=1)
    target_fft = torch.fft.rfft(target, dim=1)
    if loss_type == "mse":
        return F.mse_loss(pred_fft.abs(), target_fft.abs())
    elif loss_type == "l1":
        return F.l1_loss(pred_fft.abs(), target_fft.abs())
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


# =====================================================================
# Full Tokenizer Model V2
# =====================================================================


class ActionLatentTokenizerV2(nn.Module):
    """Action Latent Tokenizer V2.

    Loss:
      1. Recon: 모든 token (global + time + hand) → action reconstruct
      2. Hand state prediction: hand tokens → future hand states
      3. Masked latent recon: encoder output의 time token을 마스킹 후 동일 decoder로 전체 action reconstruct
      4. Global token: contrastive 또는 regression (FAST tokens → text encoder)
      5. Frequency loss (optional)

    lambda가 0인 loss의 decoder/parameter는 생성하지 않음.
    """

    def __init__(
        self,
        encoder: TimeWiseEncoder,
        recon_decoder: ReconDecoder,
        # Optional modules (None when lambda=0)
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
        # Masked latent recon config
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
    ):
        super().__init__()
        self.encoder = encoder
        self.recon_decoder = recon_decoder
        self.hand_pred_decoder = hand_pred_decoder
        self.action_text_encoder = action_text_encoder
        self.global_loss_module = global_loss_module

        self.hand_in_recon = hand_in_recon
        self.state_pred_kv_source = state_pred_kv_source

        self.lambda_recon = lambda_recon
        self.lambda_hand_pred = lambda_hand_pred
        self.lambda_mask_recon = lambda_mask_recon
        self.lambda_mask_hand_pred = lambda_mask_hand_pred
        self.lambda_global = lambda_global
        self.freq_loss_weight = freq_loss_weight
        self.mask_batch_ratio = mask_batch_ratio
        self.recon_loss_type = recon_loss_type
        self.mask_mode = mask_mode

        # Mask ratio: min/max가 설정되면 매번 [min, max]에서 균등 샘플링
        if mask_ratio_min is not None and mask_ratio_max is not None:
            self.mask_ratio_min = mask_ratio_min
            self.mask_ratio_max = mask_ratio_max
        else:
            self.mask_ratio_min = mask_ratio
            self.mask_ratio_max = mask_ratio

        self.num_global_tokens = encoder.num_global_tokens
        self.num_hand_tokens = encoder.num_hand_tokens

        # Mask token for masked latent recon
        if lambda_mask_recon > 0:
            self.mask_token = nn.Parameter(torch.randn(encoder.emb_dim))
        else:
            self.mask_token = None

        # V2 marker for wrapper detection
        self.register_buffer("_is_v2", torch.tensor(True))

    def _zero(self, device):
        return torch.tensor(0.0, device=device, requires_grad=False)

    def _recon_loss_fn(self, pred, target):
        return F.mse_loss(pred, target) if self.recon_loss_type == "mse" else F.l1_loss(pred, target)

    def forward(self, batch: dict = None, **kwargs) -> dict:
        """
        Args:
            batch: {
                "action": [B, T, D],                           (필수)
                "hand_state": [B, hand_dim],                   (hand_pred 사용 시)
                "future_hand_states": [B, num_future, hand_dim], (hand_pred 사용 시)
                "fast_tokens": [B, L],                          (global loss 사용 시)
            }
        Returns:
            dict with "loss" and per-component losses
        """
        if batch is None:
            batch = kwargs
        actions = batch["action"]  # [B, T, D]
        actions = actions.to(dtype=self.encoder.action_proj.weight.dtype)
        device = actions.device
        B, T, D = actions.shape

        Ng = self.num_global_tokens
        Nh = self.num_hand_tokens

        # --- Encode ---
        global_tok, time_tok, hand_tok = self.encoder(actions)

        # Helper: get optional tokens for decoder
        g_for_dec = global_tok if Ng > 0 else None
        h_for_dec = hand_tok if (Nh > 0 and self.hand_in_recon) else None

        # --- Loss 1: Recon ---
        if self.lambda_recon > 0:
            recons = self.recon_decoder(time_tok, global_tokens=g_for_dec, hand_tokens=h_for_dec)
            loss_recon = self._recon_loss_fn(recons, actions)
        else:
            recons = None
            loss_recon = self._zero(device)

        # --- Loss 2: Hand state prediction ---
        if self.lambda_hand_pred > 0 and self.hand_pred_decoder is not None:
            hand_state = batch["hand_state"].to(dtype=actions.dtype)  # [B, hand_dim]
            future_states = batch["future_hand_states"].to(dtype=actions.dtype)  # [B, F, hand_dim]
            kv_tokens = hand_tok if self.state_pred_kv_source == "hand" else time_tok
            pred_future = self.hand_pred_decoder(hand_state, kv_tokens)
            loss_hand_pred = F.mse_loss(pred_future, future_states)
        else:
            loss_hand_pred = self._zero(device)

        # --- Loss 3: Masked latent recon ---
        # action의 timestep을 마스킹 → encoder에 넣고 → 그 latent로 recon
        # Loss 3b (아래 블록 내부)에서 masked latent로 future state prediction도 수행
        loss_mask_hand_pred = self._zero(device)
        if self.lambda_mask_recon > 0 and self.mask_token is not None:
            num_masked = max(1, int(B * self.mask_batch_ratio))
            perm = torch.randperm(B, device=device)[:num_masked]

            # 선택된 batch의 action을 마스킹 (encoder 입력 전에 마스킹)
            masked_actions = actions[perm].clone()  # [num_masked, T, D]

            # Mask ratio 결정
            if self.mask_ratio_min == self.mask_ratio_max:
                cur_mask_ratio = self.mask_ratio_min
            else:
                cur_mask_ratio = torch.empty(1).uniform_(self.mask_ratio_min, self.mask_ratio_max).item()

            # Mask 생성: [num_masked, T]
            if self.mask_mode == "random":
                mask = torch.rand(num_masked, T, device=device) < cur_mask_ratio
            elif self.mask_mode == "block":
                num_mask = max(1, int(T * cur_mask_ratio))
                max_start = T - num_mask
                start = torch.randint(0, max_start + 1, (num_masked,), device=device)
                arange = torch.arange(T, device=device).unsqueeze(0)
                mask = (arange >= start.unsqueeze(1)) & (arange < (start + num_mask).unsqueeze(1))
            else:
                raise ValueError(f"Unknown mask_mode: {self.mask_mode}")

            # Action의 마스킹된 timestep을 mask_token으로 대체
            # mask_token: [E] → action space가 아닌 emb space이므로,
            # encoder의 action_proj 후에 마스킹해야 함
            # action_proj → pos_emb → mask → concat tokens → transformer
            masked_proj = self.encoder.action_proj(masked_actions)  # [num_masked, T, E]
            masked_proj = self.encoder.time_pos_emb(masked_proj)    # [num_masked, T, E]

            mask_expanded = mask.unsqueeze(-1).expand_as(masked_proj)  # [num_masked, T, E]
            mask_tok = self.mask_token.unsqueeze(0).unsqueeze(0).expand_as(masked_proj)
            masked_proj = torch.where(mask_expanded, mask_tok, masked_proj)

            # global/hand tokens를 prepend/append하고 transformer 통과
            parts = []
            if Ng > 0:
                parts.append(self.encoder.global_tokens.unsqueeze(0).expand(num_masked, -1, -1))
            parts.append(masked_proj)
            if Nh > 0:
                parts.append(self.encoder.hand_tokens.unsqueeze(0).expand(num_masked, -1, -1))
            masked_seq = torch.cat(parts, dim=1)
            masked_seq = self.encoder.transformer(masked_seq)

            # Split outputs
            m_global = masked_seq[:, :Ng] if Ng > 0 else None
            m_time = masked_seq[:, Ng:Ng + T]
            m_hand = masked_seq[:, Ng + T:] if (Nh > 0 and self.hand_in_recon) else None

            # 동일 decoder로 reconstruct (전체 action에 대해 loss)
            recons_masked = self.recon_decoder(m_time, global_tokens=m_global, hand_tokens=m_hand)
            loss_mask_recon = self._recon_loss_fn(recons_masked, actions[perm])

            # --- Loss 3b: Masked state prediction ---
            # masked forward의 latent로도 future state 예측 — action latent가
            # 부분 관측 하에서도 state-change 정보를 유지하도록 regularization.
            if self.lambda_mask_hand_pred > 0 and self.hand_pred_decoder is not None:
                hand_state_masked = batch["hand_state"].to(dtype=actions.dtype)  # [B, hand_dim]
                future_states_gt = batch["future_hand_states"].to(dtype=actions.dtype)  # [B, F, hand_dim]

                if self.state_pred_kv_source == "time":
                    m_kv = m_time
                elif Nh > 0:
                    # hand 토큰 chunk: hand_in_recon=False 이어도 state_pred용으로 사용 가능.
                    m_kv = masked_seq[:, Ng + T:]
                else:
                    m_kv = None

                if m_kv is not None:
                    pred_future_masked = self.hand_pred_decoder(hand_state_masked[perm], m_kv)
                    loss_mask_hand_pred = F.mse_loss(pred_future_masked, future_states_gt[perm])
        else:
            loss_mask_recon = self._zero(device)

        # --- Loss 4: Global token learning ---
        if self.lambda_global > 0 and self.action_text_encoder is not None and self.global_loss_module is not None:
            fast_tokens = batch["fast_tokens"]  # [B, L] int
            text_features = self.action_text_encoder(fast_tokens)  # [B, output_dim]
            loss_global = self.global_loss_module(global_tok, text_features)
        else:
            loss_global = self._zero(device)

        # --- Loss 5: Frequency loss ---
        if self.freq_loss_weight > 0 and recons is not None:
            loss_freq = frequency_loss(recons, actions, loss_type=self.recon_loss_type)
        else:
            loss_freq = self._zero(device)

        # --- Total ---
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

    def encode(self, actions: torch.Tensor):
        """Inference encode: [B, T, D] → (global, time, hand)"""
        return self.encoder(actions)

    def decode(self, global_tok, time_tok, hand_tok):
        """Inference decode: action reconstruct. hand_in_recon=False이면 hand_tok 무시."""
        g = global_tok if global_tok is not None and global_tok.shape[1] > 0 else None
        h = hand_tok if (hand_tok is not None and hand_tok.shape[1] > 0 and self.hand_in_recon) else None
        return self.recon_decoder(time_tok, global_tokens=g, hand_tokens=h)

    def autoencode(self, actions: torch.Tensor) -> torch.Tensor:
        """Encode → Decode: [B, T, D] → [B, T, D]"""
        g, t, h = self.encode(actions)
        return self.decode(g, t, h)
