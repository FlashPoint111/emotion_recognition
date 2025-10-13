import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from .vit_encoder import ViT_video
from .clip_model import VisionTransformer
from .lora_layers import *
from .lora_utils import *

def sup_contra_loss(logits, mask):
    logits = torch.log_softmax(logits, dim=1)
    positive_logits = logits * mask
    positive_logits_sum = torch.sum(positive_logits, dim=1) / mask.sum(dim=1)
    loss = -torch.mean(positive_logits_sum)
    return loss


'''
class AIM(nn.Module):
    def __init__(self,
                 label_num,
                 temp=0.07,
                 use_audio=True):
        super().__init__()
        self.vis_encoder = ViT_CLIP(224, 16, 16, 768, 12, 12, 0.1)
        self.vis_encoder.init_weights(pretrained='clip')
        self.fc2 = nn.Linear(768, label_num)
        for name, param in self.vis_encoder.named_parameters():
            if 'temporal_embedding' not in name and 'ln_post' not in name and 'cls_head' not in name and 'Adapter' not in name:
                param.requires_grad = False

    def forward(self, video, label=None):
        video_cls = self.vis_encoder(video)
        video_cls = video_cls.mean(dim=2)
        video_cls = self.fc2(video_cls)
        if label is not None:
            ce_loss = F.cross_entropy(video_cls, label)
            return ce_loss
        else:
            return video_cls
'''


class AIM(nn.Module):
    def __init__(self,
                 config,
                 temp=0.07,
                 use_audio=True):
        super().__init__()
        label_num = config["dataset"]["label_num"]
        self.n_frame = config["dataset"]["n_frame"]
        train_params = 0
        total_params = 0
        additional_params = 0
        self.encoder = ViT_video(config["train"]["pretrain_file"])
        self.encoder.init_weights()
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc_cls = nn.Linear(768, label_num)
        for name, param in self.encoder.named_parameters():
            param.requires_grad = False
            tmp = 1
            for num in param.shape:
                tmp *= num

            if 'ViT' in name or 'swin' in name:
                param.requires_grad = False
                total_params += tmp
                if 'cls_token' in name or 'v.norm' in name:
                    param.requires_grad = True
                    train_params += tmp

            elif 'Adapter' in name or 'norm' in name or 'video' in name or 'share_tokens' in name or 'fft' in name or 'temporal' in name:
                param.requires_grad = True
                train_params += tmp
                additional_params += tmp
                total_params += tmp
                print('########### train layer:', name, param.shape, tmp)

        print('####### Trainable params: %0.4f  #######' % (train_params * 100 / total_params))
        print(
            '####### Additional params: %0.4f  ######' % (additional_params * 100 / (total_params - additional_params)))
        print('####### Total params in M: %0.1f M  #######' % (total_params / 1000000))

    def forward(self, x, label=None):
        x = self.encoder(x)
        '''x = rearrange(x, '(b t) d -> b d t', t=self.n_frame)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.avg_pool(x)
        x = x.view(x.shape[0], -1)'''

        logits = self.fc_cls(x)
        if label is not None:
            ce_loss = F.cross_entropy(logits, label)
            return ce_loss
        else:
            return logits


class VideoClip(nn.Module):
    def __init__(self, loss_fn=None):
        super().__init__()
        self.video_encoder = VisionTransformer(224, 16, 768, 12, 8, 768)
        self.video_encoder.init_weights("ViT-B/16")
        self.video_encoder = add_lora_to_clip_vision_encoder(
            self.video_encoder,
            lora_r=8,
            lora_alpha=32,
            lora_parts=['q', 'k', 'v', 'o']
        )
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc_cls = nn.Linear(768, 11)
        for name, param in self.video_encoder.named_parameters():
            if 'temporal' in name or 'ln_post' in name:
                param.requires_grad = True
            elif 'lora_' not in name:
                param.requires_grad = False
        self.loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()

        trainable_params = list(filter(lambda p: p.requires_grad, self.parameters()))
        num_trainable = sum(p.numel() for p in trainable_params)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Trainable parameters: {num_trainable}")
        print(f"Total parameters: {total_params}")
        print(f"Trainable ratio: {100 * num_trainable / total_params:.2f}%")

    def forward(self, x, label=None):
        x, is_ordered = self.video_encoder(x)
        '''x = rearrange(x, '(b t) d -> b d t', t=16)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.avg_pool(x)
        x = x.view(x.shape[0], -1)'''
        x = self.dropout(x)
        logits = self.fc_cls(x)
        if label is not None:
            ce_loss = self.loss_fn(logits, label)
            return ce_loss
        else:
            return logits

