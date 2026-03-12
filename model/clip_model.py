from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import open_clip
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import Attention

import torch
import torch.nn as nn


class RelativeTemporalBias1D(nn.Module):
    def __init__(self, num_heads: int, max_frames: int):
        super().__init__()
        self.num_heads = num_heads
        self.max_frames = max_frames
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(2 * max_frames - 1, num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(
            self,
            query_len: int,
            key_frame_len: int,
            seeds_per_frame: int,
    ) -> torch.Tensor:
        if query_len > self.max_frames or key_frame_len > self.max_frames:
            raise ValueError(
                f"query_len={query_len}, key_frame_len={key_frame_len} exceed "
                f"max_frames={self.max_frames}"
            )
        device = self.relative_position_bias_table.device
        q_pos = torch.arange(query_len, device=device)
        k_frame_pos = torch.arange(key_frame_len, device=device).repeat_interleave(seeds_per_frame)
        rel = k_frame_pos.unsqueeze(0) - q_pos.unsqueeze(1)
        rel = rel + (self.max_frames - 1)
        bias = self.relative_position_bias_table[rel]
        bias = bias.permute(2, 0, 1).contiguous()
        return bias


class CrossPostAudioAdapter(nn.Module):
    def __init__(
            self,
            dim: int = 512,
            audio_dropout: float = 0.3,
    ):
        super().__init__()
        self.dim = dim

        self.a_norm = LayerNorm(dim)
        self.x_norm = LayerNorm(dim)

        self.gamma = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, dim)
        )

        self.beta = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, dim)
        )

        self.gate = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, dim)
        )
        self.drop = DropPath(audio_dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal(self.gamma[0].weight)
        nn.init.zeros_(self.gamma[0].bias)
        nn.init.zeros_(self.gamma[-1].weight)
        nn.init.zeros_(self.gamma[-1].bias)

        nn.init.xavier_normal(self.beta[0].weight)
        nn.init.zeros_(self.beta[0].bias)
        nn.init.zeros_(self.beta[-1].weight)
        nn.init.zeros_(self.beta[-1].bias)

        nn.init.xavier_normal(self.gate[0].weight)
        nn.init.zeros_(self.gate[0].bias)
        nn.init.xavier_normal(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

        nn.init.constant_(self.x_norm.bias, 0)
        nn.init.constant_(self.x_norm.weight, 1.0)
        nn.init.constant_(self.a_norm.bias, 0)
        nn.init.constant_(self.a_norm.weight, 1.0)


    def forward(self, sample_cls: torch.Tensor, audio: torch.Tensor):
        x0 = sample_cls

        a = audio
        gamma = self.gamma(a)
        beta = self.beta(a)
        gate = self.gate(a)

        out = x0 + torch.tanh(gate) * self.a_norm(gamma * x0 + beta)
        out = self.x_norm(out)

        return out


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)
        self.temporal_embedding = nn.Parameter(torch.zeros(1, 17, output_dim))
        self.temporal_cls = nn.Parameter(torch.zeros(1, 1, output_dim))
        trunc_normal_(self.temporal_cls, std=.02)
        nn.init.normal_(self.temporal_embedding, std=1e-6)
        self.temporal_transformer_layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.,
            batch_first=True,
            norm_first=True
        )

        self.final_norm = nn.LayerNorm(output_dim)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        self.context_rel_bias = RelativeTemporalBias1D(num_heads=8, max_frames=16)
        self.context_attn = nn.MultiheadAttention(512, 8, batch_first=True)
        self.context_alpha = nn.Parameter(torch.zeros([]))
        self.context_seed_norm = LayerNorm(output_dim)
        self.context_q_norm = LayerNorm(output_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.cross_post_adapter = CrossPostAudioAdapter(
            dim=output_dim,
            audio_dropout=0.3,
        )

    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if pretrained:
            self.pretrained = pretrained
        if isinstance(self.pretrained, str):
            self.apply(_init_weights)
            ## Load OpenAI CLIP pretrained weights
            clip_model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-16",
                pretrained="openai"
            )
            pretrain_dict = clip_model.visual.state_dict()
            self.logit_scale = clip_model.logit_scale
            del clip_model
            # del pretrain_dict['proj']
            msg = self.load_state_dict(pretrain_dict, strict=False)
            print(msg)
            torch.cuda.empty_cache()
        elif self.pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x: torch.Tensor, audio: torch.Tensor):
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        '''
        x = self.ln_post(x[:, 0, :])
        x = rearrange(x, '(b t) d -> b t d', b=B, t=T)
        '''
        x_cls_raw = self.ln_post(x[:, 0, :])
        x = torch.cat([x_cls_raw.unsqueeze(1), x[:, 1:, :]], dim=1)
        x = x @ self.proj

        x_l2 = F.normalize(x, dim=-1)
        x_cls = x[:, 0, :]
        x_patch = x[:, 1:, :]

        scores = (x_l2[:, 1:, :] * x_l2[:, 0, :].unsqueeze(1)).sum(dim=-1)
        idx = scores.topk(k=8, dim=-1, largest=True, sorted=False).indices
        seeds = torch.gather(
            x_patch,  # [B,T,N,D]
            dim=1,
            index=idx.unsqueeze(-1).expand(-1, -1, x_patch.size(-1))
        )

        seeds = self.context_seed_norm(seeds)
        x_cls = rearrange(x_cls, '(b t) d -> b t d', b=B, t=T)
        seeds = rearrange(seeds, '(b t) k d -> b (t k) d', b=B, t=T)

        rel_bias = self.context_rel_bias(
            query_len=T,
            key_frame_len=T,
            seeds_per_frame=8
        )
        rel_bias = rel_bias.repeat(B, 1, 1)
        context, _ = self.context_attn(self.context_q_norm(x_cls), seeds, seeds, attn_mask=rel_bias,
                                       average_attn_weights=False, need_weights=False)
        x = x_cls + torch.tanh(self.context_alpha) * context

        x = torch.cat([self.temporal_cls.expand(x.shape[0], -1, -1), x], dim=1)
        x = x + self.temporal_embedding
        x = self.temporal_transformer_layer(x)
        x = x[:, 0, :]
        x = self.final_norm(x)

        x = self.cross_post_adapter(x, audio)
        return x
