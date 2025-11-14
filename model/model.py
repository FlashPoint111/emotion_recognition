from .htsat import HTSAT_Swin_Transformer
from .lora_utils import *
from .vit_encoder import ViT_video
from dataclasses import dataclass
import numpy as np
import laion_clap
def sup_contra_loss(logits, mask):
    logits = torch.log_softmax(logits, dim=1)
    positive_logits = logits * mask
    positive_logits_sum = torch.sum(positive_logits, dim=1) / mask.sum(dim=1)
    loss = -torch.mean(positive_logits_sum)
    return loss
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from peft import get_peft_model, LoraConfig, TaskType

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
        # self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # self.dropout = nn.Dropout(0.5)
        # self.fc_cls = nn.Linear(768, label_num)
        self.prompt = torch.load("prompt.pt")
        self.prompt.requires_grad = False
        for name, param in self.encoder.named_parameters():
            param.requires_grad = False
            tmp = 1
            for num in param.shape:
                tmp *= num

            if 'ViT' in name or 'swin' in name:
                param.requires_grad = False
                total_params += tmp
                if 'v.norm' in name:
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
        C, _, _ = self.prompt.shape
        for i in range(C):
            prompt = self.prompt[i]

        if label is not None:
            # ce_loss = F.cross_entropy(logits, label)
            return 0
        else:
            return 0


class VideoClip(nn.Module):
    def __init__(self, loss_fn=None):
        super().__init__()
        self.video_encoder = VisionTransformer(224, 16, 768, 12, 8, 512)
        self.video_encoder.init_weights("ViT-B/16")
        self.video_encoder = add_lora_to_clip_vision_encoder(
            self.video_encoder,
            lora_r=8,
            lora_alpha=32,
            lora_parts=['q', 'k', 'v', 'o']
        )
        # self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # self.dropout = nn.Dropout(0.5)
        # self.fc_cls = nn.Linear(768, 11)
        prompt = torch.load("./dataset/prompts.pt")
        self.prompt = nn.Parameter(prompt)
        self.prompt.requires_grad = False
        for name, param in self.video_encoder.named_parameters():
            if 'temporal' in name or 'ln_post' in name or 'final' in name:
                param.requires_grad = True
            elif 'lora_' not in name:
                param.requires_grad = False
        self.loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()
        self.logit_scale = nn.Parameter(self.video_encoder.logit_scale)
        # self.logit_scale.requires_grad = True
        '''
        self.loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss(label_smoothing=0.1)
        #self.register_buffer("prompt", prompt)
        C, K, _ = self.prompt.shape
        '''

    def forward(self, x, label=None):
        x_proj, x_feat = self.video_encoder(x)
        x = F.normalize(x_proj, dim=-1)
        similarity = (self.logit_scale.exp() * x @ self.prompt.transpose(0, 1))
        if label is not None:
            ce_loss = self.loss_fn(similarity, label)
            if not self.training:
                return similarity, ce_loss
            else:
                return ce_loss
        else:
            return similarity


@dataclass
class CLAPAudioCfp:
    model_type: str = "HTSAT"
    model_name: str = "tiny"
    sample_rate: int = 48000
    # Param
    audio_length: int = 1024
    window_size: int = 1024
    hop_size: int = 1024
    fmin: int = 50
    fmax: int = 14000
    class_num: int = 527
    mel_bins: int = 64
    clip_samples: int = 480000


class HTSAT(nn.Module):
    def __init__(self, loss_fn=None):
        super().__init__()
        audio_cfg = {
            "audio_length": 1024,
            "clip_samples": 480000,
            "mel_bins": 64,
            "sample_rate": 48000,
            "window_size": 1024,
            "hop_size": 480,
            "fmin": 50,
            "fmax": 14000,
            "class_num": 527,
            "model_type": "HTSAT",
            "model_name": "tiny"
        }
        audio_cfg = CLAPAudioCfp(**audio_cfg)
        self.audio_encoder = HTSAT_Swin_Transformer(
            spec_size=256,
            patch_size=4,
            patch_stride=(4, 4),
            num_classes=11,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[4, 8, 16, 32],
            window_size=8,
            config=audio_cfg,
            enable_fusion=False,
            fusion_type=None
        )
        model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-tiny')
        model.load_ckpt('./630k-audioset-best.pt')
        self.audio_projection = model.model.audio_projection
        pretrain = torch.load('./630k-audioset-best.pt')
        pretrain = pretrain['state_dict']
        prefix_to_remove = 'module.audio_branch.'
        len_prefix = len(prefix_to_remove)
        audio_branch_state_dict = {}
        for key, value in pretrain.items():
            if key.startswith(prefix_to_remove):
                new_key = key[len_prefix:]
                audio_branch_state_dict[new_key] = value

        msg = self.audio_encoder.load_state_dict(audio_branch_state_dict, strict=False)
        print(msg)
        lora_r, lora_alpha, lora_dropout = 8, 32, 0.1
        for layer in self.audio_encoder.layers:
            for block in layer.blocks:
                qkv = LinearLoRA(block.attn.qkv, lora_r, lora_alpha, False, lora_dropout)
                proj = LinearLoRA(block.attn.proj, lora_r, lora_alpha, False, lora_dropout)
                fc1 = LinearLoRA(block.mlp.fc1, lora_r, lora_alpha, False, lora_dropout)
                fc2 = LinearLoRA(block.mlp.fc2, lora_r, lora_alpha, False, lora_dropout)
                block.attn.qkv = qkv
                block.attn.proj = proj
                block.mlp.fc1 = fc1
                block.mlp.fc2 = fc2

        for name, param in self.named_parameters():
            if 'lora_' not in name:
                param.requires_grad = False
        '''self.fc_cls = nn.Linear(1024, 11)
        self.dropout = nn.Dropout(0.5)'''
        # self.loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss(label_smoothing=0.1)
        '''trainable_params = list(filter(lambda p: p.requires_grad, self.parameters()))
        num_trainable = sum(p.numel() for p in trainable_params)
        total_params = sum(p.numel() for p in self.parameters())'''
        audio_prompt = torch.load("./dataset/audio_prompt.pt")
        self.audio_prompt = nn.Parameter(audio_prompt)
        self.logit_scale = nn.Parameter(pretrain['module.logit_scale_a'])
        C, K, _ = self.audio_prompt.shape
        self.alpha = nn.Parameter(torch.ones((C, K)) / K)
        self.final_norm = nn.LayerNorm(768)
        '''print(f"Trainable parameters: {num_trainable}")
        print(f"Total parameters: {total_params}")
        print(f"Trainable ratio: {100 * num_trainable / total_params:.2f}%")'''
        print(sum(p.numel() for p in self.audio_encoder.parameters()))
        print('-----------------------------------------------------')
        del model, pretrain, audio_branch_state_dict

    def forward(self, x, mixup_lambda=None, label=None):
        x = self.audio_encoder(x, mixup_lambda=mixup_lambda)
        x_feat = self.final_norm(x)
        x = self.audio_projection(x)
        x = F.normalize(x, dim=-1)
        audio_prompt_norm = F.normalize(self.audio_prompt, dim=-1)
        similarity = (self.logit_scale.exp() * audio_prompt_norm @ x.transpose(-2, -1)).permute(2, 0, 1)
        logits = (similarity * self.alpha).sum(dim=-1)
        return logits, x_feat

        '''x = self.dropout(x)
        logits = self.fc_cls(x)
        if label is not None:
            ce_loss = self.loss_fn(logits, label)
            return ce_loss
        else:
            return logits'''


class CLAIP(nn.Module):
    def __init__(self, loss_fn=None):
        super().__init__()
        self.video_encoder = VideoClip()
        self.audio_encoder = HTSAT()
        self.loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()

        self.Pv = nn.Linear(768, 64, bias=False)
        self.Pa = nn.Linear(768, 64, bias=False)
        self.Wu = nn.Linear(64, 11, bias=False)
        self.Ws = nn.Linear(768 + 768, 11, bias=False)
        nn.init.xavier_uniform_(self.Pv.weight)
        nn.init.xavier_uniform_(self.Pa.weight)
        nn.init.xavier_uniform_(self.Wu.weight)
        nn.init.xavier_uniform_(self.Ws.weight)
        self.b = nn.Parameter(torch.zeros(11))

        '''self.x_norm = nn.LayerNorm(768)
        self.a_norm = nn.LayerNorm(768)
        self.DropPath = DropPath(0.1)
        nn.init.constant_(self.x_norm.bias, 0)
        nn.init.constant_(self.x_norm.weight, 1.0)
        nn.init.constant_(self.a_norm.bias, 0)
        nn.init.constant_(self.a_norm.weight, 1.0)'''

        trainable_params = list(filter(lambda p: p.requires_grad, self.parameters()))
        num_trainable = sum(p.numel() for p in trainable_params)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Trainable parameters: {num_trainable}")
        print(f"Total parameters: {total_params}")
        print(f"Trainable ratio: {100 * num_trainable / total_params:.2f}%")
        print("--------------------------------------------------------")

    def forward(self, x, a, label=None, mixup_lambda=None):
        x, x_feat = self.video_encoder(x)
        a, a_feat = self.audio_encoder(a, mixup_lambda=mixup_lambda)
        x = x - x.mean(dim=-1, keepdim=True)
        a = a - a.mean(dim=-1, keepdim=True)
        joint = torch.cat([x_feat, a_feat], dim=-1)
        u = self.Pv(x_feat)
        v = self.Pa(a_feat)
        r = u * v
        s = self.Wu(r) + self.Ws(joint) + self.b
        alpha = torch.sigmoid(s)

        with torch.no_grad():
            out_v = torch.log_softmax(x, dim=-1)
            out_a = torch.log_softmax(a, dim=-1)
            H_v = torch.sum(-out_v.exp() * out_v, dim=1).mean()
            H_a = torch.sum(-out_a.exp() * out_a, dim=1).mean()
        '''
        conf = torch.softmax(torch.stack([H_v, H_a], dim=-1), dim=-1)
        w_x, w_a = conf[..., 0], conf[..., 1]

        def logit(p):
            return torch.log(p.clamp_min(1e-6)) - torch.log1p(-p.clamp_min(1e-6))

        eps = 1e-6
        logit_alpha_feat = logit(alpha)
        logit_alpha_conf = torch.log(w_x + eps) - torch.log(w_a + eps)
        logit_alpha = logit_alpha_feat.squeeze() + logit_alpha_conf
        alpha = torch.sigmoid(logit_alpha).unsqueeze(-1)'''
        logits = alpha * x + (1 - alpha) * a

        if label is not None:
            ce_loss_multi = self.loss_fn(logits, label)
            ce_loss_x = self.loss_fn(x, label)
            ce_loss_a = self.loss_fn(a, label)
            if self.training:
                return ce_loss_multi, ce_loss_x, ce_loss_a, H_v, H_a
            else:
                return logits, ce_loss_multi, ce_loss_x, ce_loss_a, H_v, H_a
        else:
            return logits
