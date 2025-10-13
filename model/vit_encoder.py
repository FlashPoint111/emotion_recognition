import torch
import torch.nn as nn
from einops import rearrange

from .custom_vit import my_vit
from timm.models.layers import DropPath, trunc_normal_
from timm.models.vision_transformer import Attention

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
        self.MLP_Adapter = nn.ModuleList(
            [Adapter(d_model, mlp_ratio=0.25, skip_connect=False) for d_model in hidden_list])

        self.video_temporal_embedding = nn.Parameter(torch.zeros(1, 17, 768) * .02)
        self.scale = 768 ** -0.5
        self.drop_path = DropPath(0.2)
        self.video_cls = nn.Parameter(torch.randn(1, 1, 768) * .02)
        self.video_attn = Attention(768)


    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)
        # initialize S_Adapter
        for n, m in self.named_modules():
            if 'S_Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)
        # initialize MLP_Adapter
        for n, m in self.named_modules():
            if 'MLP_Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

    def forward(self, x):
        # x: video, size = [B, T, C, H, W]
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x, x_shape = self.ViT.forward_patch(x, is_shape_info=True)
        _, nx, _ = x.shape

        for idx_layer, blk in enumerate(self.ViT.v.blocks):
            x_attn = self.S_Adapter[idx_layer](blk.drop_path1(blk.ls1(blk.attn(blk.norm1(x)))))
            x = x + x_attn

            x_norm = blk.norm2(x)
            x_ffn = blk.drop_path2(blk.ls2(blk.mlp(x_norm)))
            x = x + x_ffn + self.drop_path(self.MLP_Adapter[idx_layer](x_norm))

        x = self.ViT.v.norm(x)
        x_cls = x[:, 0, :]
        x_cls = rearrange(x_cls, '(b t) d -> b t d', t=T)
        x_cls = torch.cat([self.video_cls.expand(x_cls.shape[0], -1, -1), x_cls], dim=1)
        x_cls = x_cls + self.video_temporal_embedding
        x_cls = self.video_attn(x_cls)
        v_cls = x_cls[:, 0, :]

        return v_cls
