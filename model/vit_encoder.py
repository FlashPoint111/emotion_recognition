import math

import torch
import torch.nn as nn
from einops import rearrange
import numpy as np
from collections import OrderedDict
from .custom_vit import my_vit
from timm.models.layers import PatchEmbed, trunc_normal_
from timm.models.vision_transformer import _cfg, Attention, DropPath, Mlp, partial, LayerScale, _cfg, Block
# from .sparsemax import Sparsemax
import torch.nn.functional as F

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25,
                 act_layer=nn.GELU, skip_connect=True,
                 attention=True,
                 num_heads=8, qkv_bias=False, attn_drop=0., drop=0.):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        self.attn = Attention(D_hidden_features, num_heads=num_heads,
                              qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop) if attention else nn.Identity()

        self.apply(self._init_weights)
        nn.init.constant_(self.D_fc2.weight, 0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.attn(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TMAdapter(nn.Module):
    def __init__(self, D_features, num_frames, ratio=0.25):
        super().__init__()
        self.num_frames = num_frames
        self.T_Adapter = Adapter(
            D_features, mlp_ratio=ratio, skip_connect=False, attention=True)
        self.norm = nn.LayerNorm(D_features)
        self.S_Adapter = Adapter(
            D_features, mlp_ratio=ratio, skip_connect=False, attention=False)

    def forward(self, x):
        # x is (BT, HW+1, D)
        bt, n, d = x.shape
        ## temporal adaptation
        xt = rearrange(x, '(b t) n d -> (b n) t d', t=self.num_frames)
        xt = self.T_Adapter(xt)
        x = rearrange(xt, '(b n) t d -> (b t) n d', n=n)

        x = self.S_Adapter(self.norm(x))
        return x


class TemporalConvAdapter(nn.Module):
    """Depthwise 1D Conv along time with tiny pointwise mixing.
       Input:  (B*N, T, D) ; Output: same shape
    """
    def __init__(self, d_model, ksize=3, dilation=2, expand=0.5, drop=0.0, skip_connect=True):
        super().__init__()
        self.dw = nn.Conv1d(d_model, d_model, kernel_size=ksize, padding=dilation*(ksize//2),
                            dilation=dilation, groups=d_model, bias=True)
        hid = max(8, int(d_model * expand))
        self.D_fc1 = nn.Linear(d_model, hid)
        self.act = nn.GELU()
        self.D_fc2 = nn.Linear(hid, d_model)
        self.drop = nn.Dropout(drop)
        self.skip_connect = skip_connect
        nn.init.constant_(self.D_fc2.weight, 0)
        nn.init.constant_(self.D_fc2.bias, 0)

    def forward(self, x):     # x: (B*N, T, D)
        y = x.transpose(1, 2)         # (B*N, D, T)
        y = self.dw(y).transpose(1, 2) + x     # (B*N, T, D)
        y = self.D_fc2(self.act(self.D_fc1(y)))
        if self.skip_connect:
            return x + self.drop(y)
        else:
            return self.drop(y)


def stft_3d(x, temporal_window_size=5, temporal_stride=1):
    B, C, T, H, W = x.shape
    padding_t = (temporal_window_size - 1) // 2
    padded_x = F.pad(x, (0, 0, 0, 0, padding_t, padding_t), mode='reflect')
    windows = padded_x.unfold(2, temporal_window_size, temporal_stride)

    # [B, C, T_out, H, W, win_t] -> [B, C, T_out, win_t, H, W]
    windows = windows.permute(0, 1, 2, 5, 3, 4)
    stft_result = torch.fft.fftn(windows, dim=(-3, -2, -1))

    # [B, C, T_out, win_t, H, W] -> [B, T_out, win_t, H, W, C]
    stft_result = stft_result.permute(0, 2, 3, 4, 5, 1)
    stft_result = torch.fft.fftshift(stft_result, dim=(2, 3, 4))
    stft_result = torch.abs(stft_result)
    return torch.log1p(stft_result)


class Prompt_block(nn.Module):
    def __init__(self, inplanes=None, smooth=False, num_frames=1, ratio=0.25):
        super(Prompt_block, self).__init__()

        self.num_frames = num_frames
        hide_channel = int(inplanes * ratio)
        self.D_fc1 = nn.Linear(inplanes, hide_channel)
        self.D_fc2 = nn.Linear(inplanes, hide_channel)
        self.D_fc3 = nn.Linear(hide_channel, inplanes)
        self.TMA = TMAdapter(inplanes, num_frames, ratio=ratio)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, xs, x_fft):
        """ Forward pass with input x. """

        xt = self.TMA(xs)

        xs = self.D_fc1(xs)
        x_fft = self.D_fc2(x_fft)
        d_k = xs.size(-1)
        qk_weight = torch.bmm(xs, x_fft.transpose(-2, -1))  # math.sqrt(d_k)
        qk_weight = F.softmax(qk_weight, dim=-1)
        xm = torch.bmm(qk_weight, x_fft)

        xm = self.D_fc3(xm)
        return xt, xm



class ViT_video(nn.Module):
    def __init__(self, n_frame, pretrain_file):
        super(ViT_video, self).__init__()
        self.ViT = my_vit(name='vit_base_patch16_224_in21k', pretrain_file=pretrain_file)
        del self.ViT.v.head
        self.n_frame = n_frame

        hidden_list = []
        for idx_layer, block in enumerate(self.ViT.v.blocks):
            hidden_d_size = block.mlp.fc1.in_features
            hidden_list.append(hidden_d_size)

        self.prompt_block = nn.ModuleList([Prompt_block(d_model, num_frames=16) for d_model in hidden_list])
        self.prompt_x_norm = nn.ModuleList([nn.LayerNorm(d_model) for d_model in hidden_list])
        self.prompt_fft_norm = nn.ModuleList([nn.LayerNorm(d_model) for d_model in hidden_list])

        # self.video_temporal_embedding = nn.Embedding(self.n_frame, 768)
        # self.temporal_transformer = TemporalTransformer(width=768, layers=4, heads=12)
        self.video_temporal_embedding = nn.Parameter(torch.zeros(1, self.n_frame, 768))
        self.scale = 768 ** -0.5
        self.shuffle = 0.5
        self.fft_window = 5
        self.fft_proj = nn.Conv2d(self.fft_window * 3, 768, 16, 16)
        self.cls_token = self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, x):
        # x: video, size = [B, T, C, H, W]
        B, C, T, H, W = x.shape
        with torch.no_grad():
            x_fft = stft_3d(x, self.fft_window)
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        # x, x_shape = self.ViT.forward_patch(x, is_shape_info=True)
        x = self.ViT.v.patch_embed(x)

        x_fft = rearrange(x_fft, 'b t i h w c -> (b t) (i c) h w')
        x_fft = self.fft_proj(x_fft)
        x_fft = x_fft.flatten(-2).transpose(-1, -2)

        xt, xm = self.prompt_block[0](self.prompt_x_norm[0](x), self.prompt_fft_norm[0](x_fft))
        x = x + xt + xm

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.ViT.v.pos_embed
        _, nx, _ = x.shape
        x = rearrange(x, '(b t) n d -> (b n) t d', t=T)
        x = x + self.video_temporal_embedding
        x = rearrange(x, '(b n) t d -> (b t) n d', t=T, n=nx)
        '''
        shuffle_flag = torch.zeros((B, T)).to(x.device)
        if self.training:
            x = rearrange(x, '(b n) t d -> b n t d', t=T, n=nx)
            x_shuffled = torch.zeros_like(x)
            for i in range(B):
                if torch.rand(1).item() > self.shuffle:
                    shuffled_indices = torch.randperm(T).to(x.device)
                    x_shuffled[i, :, :, :] = x[i, :, shuffled_indices, :]
                    shuffle_flag[i, :] = shuffled_indices
                else:
                    x_shuffled[i, :, :, :] = x[i, :, :, :]
                    shuffle_flag[i, :] = torch.arange(T)
            x = x_shuffled
            del x_shuffled
            x = rearrange(x, 'b n t d -> (b t) n d', t=T, n=nx)
        else:
            x = rearrange(x, '(b n) t d -> (b t) n d', t=T, n=nx)'''

        for idx_layer, blk in enumerate(self.ViT.v.blocks):
            if idx_layer >= 1:
                x_ori = x
                x = self.prompt_x_norm[idx_layer](x)
                xm = self.prompt_fft_norm[idx_layer](xm)
                xt, xm = self.prompt_block[idx_layer](x[:, 1:], xm)
                x = torch.cat([x_ori[:, :1], x_ori[:, 1:] + xt + xm], dim=1)

            '''xt = rearrange(x, '(b t) n d -> (b n) t d', t=T)
            xt = self.T_Adapter[idx_layer](blk.drop_path1(blk.ls1(blk.attn(blk.norm1(xt)))))
            xt = rearrange(xt, '(b n) t d -> (b t) n d', n=nx)
            x = x + self.drop_path(xt)'''

            x_attn = blk.drop_path1(blk.ls1(blk.attn(blk.norm1(x))))
            x = x + x_attn

            x_ffn = blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(x))))
            x = x + x_ffn

        x = self.ViT.v.norm(x)
        x_cls = x[:, 0, :]
        x_cls = rearrange(x_cls, '(b t) d -> b t d', t=self.n_frame)

        '''seq_length = T
        position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand(x_cls.size(0), -1)
        video_temporal_embedding = self.video_temporal_embedding(position_ids)
        x_temporal = x_cls + video_temporal_embedding
        x_temporal = self.temporal_transformer(x_temporal.transpose(0, 1)).transpose(0, 1)

        x_cls = x_cls + x_temporal'''

        return x_cls
