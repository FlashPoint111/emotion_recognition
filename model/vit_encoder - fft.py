import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .custom_vit import my_vit
from timm.models.vision_transformer import _cfg, Attention, DropPath, Mlp, partial, LayerScale, _cfg, Block
from torchvision.models.resnet import Bottleneck
from torchvision.transforms.functional import rgb_to_grayscale
# from .sparsemax import Sparsemax
import math


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


class TMAdapter(nn.Module):
    def __init__(self, D_features, num_frames, ratio=0.25):
        super().__init__()
        self.num_frames = num_frames
        self.T_Adapter = Adapter(
            D_features, mlp_ratio=ratio, skip_connect=False, attention=True)
        self.norm = nn.LayerNorm(D_features)
        self.S_Adapter = Adapter(
            D_features, mlp_ratio=ratio, skip_connect=False, attention=False)
        self.scale = nn.Parameter(torch.ones(1) * 0.1)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x is (BT, HW+1, D)
        bt, n, d = x.shape
        ## temporal adaptation
        xt = rearrange(x, '(b t) n d -> (b n) t d', t=self.num_frames)
        xt = self.T_Adapter(xt)
        xt = rearrange(xt, '(b n) t d -> (b t) n d', n=n)
        x = x + self.scale * xt

        x = self.S_Adapter(self.norm(x))
        return x


class fft_encoder(nn.Module):
    def __init__(self, pyramid_channels=64, out_channels=256):
        super(fft_encoder, self).__init__()
        self.lateral_conv1 = nn.Conv2d(1, pyramid_channels, kernel_size=1)
        self.lateral_conv2 = nn.Conv2d(1, pyramid_channels, kernel_size=1)
        self.lateral_conv3 = nn.Conv2d(1, pyramid_channels, kernel_size=1)
        self.lateral_conv4 = nn.Conv2d(1, pyramid_channels, kernel_size=1)
        self.fusion_smooth = nn.Conv2d(pyramid_channels, pyramid_channels, kernel_size=3, padding=1)
        downsample1 = nn.Sequential(
            nn.Conv2d(pyramid_channels, 256, kernel_size=1, stride=2),
            nn.BatchNorm2d(256),
        )
        downsample2_4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=2),
            nn.BatchNorm2d(256),
        )
        self.downsample_backbone = nn.Sequential(
            Bottleneck(pyramid_channels, 64, stride=2, downsample=downsample1),
            Bottleneck(256, 64, stride=2, downsample=downsample2_4),
            Bottleneck(256, 64, stride=2, downsample=downsample2_4),
            Bottleneck(256, 64, stride=2, downsample=downsample2_4)
        )

    def forward(self, x1, x2, x3, x4):
        p4 = self.lateral_conv4(x4)
        p3 = self.lateral_conv3(x3) + p4
        p2 = self.lateral_conv2(x2) + p3
        p1 = self.lateral_conv1(x1) + p2

        fused_feature = self.fusion_smooth(p1)

        final_feature_map = self.downsample_backbone(fused_feature)

        return final_feature_map


class ViT_video(nn.Module):
    def __init__(self, pretrain_file):
        super(ViT_video, self).__init__()
        self.ViT = my_vit(name='vit_base_patch16_224_in21k', pretrain_file=pretrain_file)
        del self.ViT.v.head

        hidden_list = []
        for idx_layer, block in enumerate(self.ViT.v.blocks):
            hidden_d_size = block.mlp.fc1.in_features
            hidden_list.append(hidden_d_size)

        self.ST_Adapter = nn.ModuleList(
            [TMAdapter(d_model, 16, ratio=0.25) for d_model in hidden_list])
        ''' self.T_Adapter = nn.ModuleList(
            [Adapter(d_model, mlp_ratio=0.25, skip_connect=False) for d_model in hidden_list])
        self.MLP_Adapter = nn.ModuleList(
            [Adapter(d_model, mlp_ratio=0.25, skip_connect=False) for d_model in hidden_list])'''
        self.fft_dw = nn.ModuleList([nn.Linear(768, 256) for _ in hidden_list])
        self.fft_up = nn.ModuleList([nn.Linear(256, 768) for _ in hidden_list])

        self.video_temporal_embedding = nn.Parameter(torch.zeros(1, 16, 768))
        self.scale = 768 ** -0.5

        self.fft_window = 5
        self.fft_proj = fft_encoder()
        # self.fft_scale = nn.Parameter(torch.zeros(12))
        self.drop_path = DropPath(0.2)

    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)
        # initialize S_Adapter
        '''for n, m in self.named_modules():
            if 'Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)'''

    def forward(self, x, x_fft):
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x, x_shape = self.ViT.forward_patch(x, is_shape_info=True)
        _, nx, _ = x.shape
        x_fft = rearrange(x_fft, 'b t c h w ->  c (b t) h w')
        f4, f8, f16, f32 = x_fft.unsqueeze(2)
        x_fft = self.fft_proj(f4, f8, f16, f32)
        x_fft = x_fft.flatten(2).transpose(1, 2)

        x = rearrange(x, '(b t) n d -> (b n) t d', t=T)
        x = x + self.video_temporal_embedding
        x = rearrange(x, '(b n) t d -> (b t) n d', n=nx)

        x_dw = self.fft_dw[0](x[:, 1:, :])
        x_weight = torch.softmax(x_dw, dim=1)
        xm = x_weight * x_dw
        x_fft = xm + x_fft
        x_up = self.fft_up[0](x_fft)
        xt = self.ST_Adapter[0](x[:, 1:, :])
        x[:, 1:, :] = x[:, 1:, :] + x_up + xt

        for idx_layer, blk in enumerate(self.ViT.v.blocks):
            if idx_layer >= 1:
                x_dw = self.fft_dw[idx_layer](x[:, 1:, :])
                x_weight = torch.softmax(x_dw, dim=1)
                xm = x_weight * x_dw
                x_fft = xm + x_fft
                x_up = self.fft_up[idx_layer](x_fft)
                xt = self.ST_Adapter[idx_layer](x[:, 1:, :])
                x[:, 1:, :] = x[:, 1:, :] + x_up + xt

            x_attn = blk.drop_path1(blk.ls1(blk.attn(blk.norm1(x))))
            x = x + x_attn

            x_ffn = blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(x))))
            x = x + x_ffn

        x = self.ViT.v.norm(x)
        x_cls = x[:, 0, :]
        return x_cls
