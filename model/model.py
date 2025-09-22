import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from .vit_encoder import ViT_video


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
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc_cls = nn.Linear(768, label_num)
        self.dw = nn.Conv1d(768, 768, 3, padding=1, groups=768, bias=True)
        self.shuffle_cls = nn.Linear(768*self.n_frame, 1)
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

            elif 'Adapter' in name or 'norm' in name or 'conv3d' in name or 'share_tokens' in name or 'fdt' in name or 'temporal' in name:
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
        x, shuffle_flag = self.encoder(x)
        x = rearrange(x, '(b t) d -> b d t', t=self.n_frame)
        x_shuffle = rearrange(x, 'b d t -> b (t d)', t=self.n_frame, d=768)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.avg_pool(x)
        x = x.view(x.shape[0], -1)

        logits = self.fc_cls(x)
        if label is not None:
            shuffle_logits = self.shuffle_cls(x_shuffle)
            ce_loss = F.cross_entropy(logits, label)
            shuffle_loss = F.binary_cross_entropy_with_logits(shuffle_logits, shuffle_flag.unsqueeze(-1))
            return ce_loss + shuffle_loss
        else:
            return logits
