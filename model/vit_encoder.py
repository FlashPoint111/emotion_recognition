import torch
import torch.nn as nn
from einops import rearrange
import numpy as np

from .custom_vit import my_vit


# from .sparsemax import Sparsemax


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        nn.init.constant_(self.D_fc2.weight, 0)
        nn.init.constant_(self.D_fc2.bias, 0)

    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
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


class ViT_video(nn.Module):
    def __init__(self, pretrain_file):
        super(ViT_video, self).__init__()
        self.ViT = my_vit(name='vit_base_patch16_224_in21k', pretrain_file=pretrain_file)
        del self.ViT.v.head

        hidden_list = []
        for idx_layer, block in enumerate(self.ViT.v.blocks):
            hidden_d_size = block.mlp.fc1.in_features
            hidden_list.append(hidden_d_size)

        self.S_Adapter = nn.ModuleList(
            [Adapter(d_model, mlp_ratio=0.25) for d_model in hidden_list])
        self.T_Adapter = nn.ModuleList([
            Adapter(d_model, mlp_ratio=0.25, skip_connect=False) for d_model in hidden_list
        ])
        self.MLP_Adapter = nn.ModuleList(
            [Adapter(d_model, mlp_ratio=0.25, skip_connect=False) for d_model in hidden_list])

        self.video_temporal_embedding = nn.Parameter(torch.zeros(1, 16, 768))
        self.scale = 768 ** -0.5
        self.shuffle = 0.5

    def forward(self, x):
        # x: video, size = [B, T, C, H, W]
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x, x_shape = self.ViT.forward_patch(x, is_shape_info=True)
        _, nx, _ = x.shape
        x = rearrange(x, '(b t) n d -> (b n) t d', t=T)
        x = x + self.video_temporal_embedding
        shuffle_flag = torch.zeros(B).to(x.device)
        if self.training:
            x = rearrange(x, '(b n) t d -> b n t d', t=T, n=nx)
            x_shuffled = torch.zeros_like(x)
            for i in range(B):
                if torch.rand(1).item() > self.shuffle:
                    shuffled_indices = torch.randperm(T).to(x.device)
                    x_shuffled[i, :, :, :] = x[i, :, shuffled_indices, :]
                    shuffle_flag[i] = 1
                else:
                    x_shuffled[i, :, :, :] = x[i, :, :, :]
            x = x_shuffled
            del x_shuffled
            x = rearrange(x, 'b n t d -> (b t) n d', t=T, n=nx)
        else:
            x = rearrange(x, '(b n) t d -> (b t) n d', t=T, n=nx)


        for idx_layer, blk in enumerate(self.ViT.v.blocks):
            xt = rearrange(x, '(b t) n d -> (b n) t d', t=T)
            xt = self.T_Adapter[idx_layer](blk.drop_path1(blk.ls1(blk.attn(blk.norm1(xt)))))
            xt = rearrange(xt, '(b n) t d -> (b t) n d', n=nx)
            x = x + xt

            x_attn = self.S_Adapter[idx_layer](blk.drop_path1(blk.ls1(blk.attn(blk.norm1(x)))))
            x = x + x_attn

            x_ffn = blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(x))))
            x = x + x_ffn + self.MLP_Adapter[idx_layer](x_ffn)

        x = self.ViT.v.norm(x)
        x_cls = x[:, 0, :]
        return x_cls, shuffle_flag
