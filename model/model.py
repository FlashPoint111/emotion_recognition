from dataclasses import dataclass

import laion_clap
import numpy as np
import open_clip

from .clip_model import Transformer, LayerNorm
from .htsat import HTSAT_Swin_Transformer
from .lora_utils import *


def sup_contra_loss(logits, mask):
    logits = torch.log_softmax(logits, dim=1)
    positive_logits = logits * mask
    positive_logits_sum = torch.sum(positive_logits, dim=1) / mask.sum(dim=1)
    loss = -torch.mean(positive_logits_sum)
    return loss


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

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        self.audio_encoder.load_state_dict(audio_branch_state_dict, strict=False)

        lora_r, lora_alpha, lora_dropout = 4, 8, 0.1
        for layer in self.audio_encoder.layers:
            for block in layer.blocks:
                qkv = LinearLoRA(block.attn.qkv, lora_r, lora_alpha, False, lora_dropout)
                # proj = LinearLoRA(block.attn.proj, lora_r, lora_alpha, False, lora_dropout)
                # fc1 = LinearLoRA(block.mlp.fc1, lora_r, lora_alpha, False, lora_dropout)
                # fc2 = LinearLoRA(block.mlp.fc2, lora_r, lora_alpha, False, lora_dropout)
                block.attn.qkv = qkv
                # block.attn.proj = proj
                # block.mlp.fc1 = fc1
                # block.mlp.fc2 = fc2
        for name, param in self.named_parameters():
            if 'lora_' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        self.audio_norm1 = nn.LayerNorm(512)
        self.audio_norm2 = nn.LayerNorm(512)
        self.fc1 = nn.Linear(512, 64)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(64, 512)
        nn.init.constant_(self.fc2.weight, 0)
        nn.init.constant_(self.fc2.bias, 0)
        del model, pretrain, audio_branch_state_dict

    def forward(self, x, mixup_lambda=None):
        x = self.audio_encoder(x, mixup_lambda=mixup_lambda)
        x = self.audio_projection(x)
        x = self.audio_norm1(x)
        xs = self.fc1(x)
        xs = self.fc2(self.act(xs))
        x = xs + x
        return self.audio_norm2(x)


class CLAIP(nn.Module):
    def __init__(self, loss_fn):
        super().__init__()
        self.context_length = 77
        self.visual = VisionTransformer(224, 16, 768, 12, 8, 512)
        self.transformer = Transformer(
            width=512,
            layers=12,
            heads=8,
            attn_mask=self.build_attention_mask()
        )
        self.vocab_size = 49408
        self.token_embedding = nn.Embedding(49408, 512)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, 512))
        self.ln_final = LayerNorm(512)

        self.text_projection = nn.Parameter(torch.empty(512, 512))
        self.loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()

        self.initialize_clip_parameters()
        self.apply_lora()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_a = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        for name, param in self.named_parameters():
            if 'temporal' in name or 'logit_scale' in name or 'final_norm' in name or 'context' in name or 'cross' in name:
                param.requires_grad = True
            elif 'lora_' not in name:
                param.requires_grad = False

        self.gate = nn.Linear(1024, 2)
        self.kl_loss = nn.KLDivLoss()
        n_ctx = 8
        ctx_dim = 512
        tokenizer = open_clip.get_tokenizer('ViT-B-16')
        classnames = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise", "contempt", "anxiety",
                      "helplessness",
                      "disappointment"]
        n_cls = len(classnames)
        p_ctx_vectors = torch.empty(n_ctx, ctx_dim)
        n_ctx_vectors = torch.empty(n_ctx, ctx_dim)
        nn.init.normal_(p_ctx_vectors, std=0.02)
        nn.init.normal_(n_ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx)

        self.p_ctx = nn.Parameter(p_ctx_vectors)
        self.n_ctx = nn.Parameter(n_ctx_vectors)
        p_prompts = [prompt_prefix + " " + name + "." for name in classnames]
        n_prompts = [prompt_prefix + " no " + name + "." for name in classnames]
        self.p_tokenized_prompts = torch.cat([tokenizer(p) for p in p_prompts])
        self.n_tokenized_prompts = torch.cat([tokenizer(p) for p in n_prompts])

        with torch.no_grad():
            p_embedding = self.token_embedding(self.p_tokenized_prompts)
            n_embedding = self.token_embedding(self.n_tokenized_prompts)

        self.register_buffer("token_prefix", p_embedding[:, :1, :])  # SOS
        self.register_buffer("p_token_suffix", p_embedding[:, 1 + n_ctx:, :])  # CLS, EOS
        self.register_buffer("n_token_suffix", n_embedding[:, 1 + n_ctx:, :])

        self.audio_encoder = HTSAT()

        trainable_params = list(filter(lambda p: p.requires_grad, self.parameters()))
        num_trainable = sum(p.numel() for p in trainable_params)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Trainable parameters: {num_trainable}")
        print(f"Total parameters: {total_params}")
        print(f"Trainable ratio: {100 * num_trainable / total_params:.2f}%")
        print("--------------------------------------------------------")

    def initialize_clip_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16",
            pretrained="openai"
        )
        pretrain_dict = clip_model.state_dict()
        del clip_model
        msg = self.load_state_dict(pretrain_dict, strict=False)
        print(msg)
        torch.cuda.empty_cache()

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def apply_lora(self):
        for i, block in enumerate(self.transformer.resblocks):
            for name, submodule in block.named_children():
                if isinstance(submodule, nn.MultiheadAttention):
                    new_multi_head_lora = PlainMultiheadAttentionLoRA(
                        submodule, enable_lora=['q', 'k', 'v'], r=4, lora_alpha=8,
                        dropout_rate=0.1)
                    setattr(block, name, new_multi_head_lora)
        for i, block in enumerate(self.visual.transformer.resblocks):
            for name, submodule in block.named_children():
                if isinstance(submodule, nn.MultiheadAttention):
                    new_multi_head_lora = PlainMultiheadAttentionLoRA(
                        submodule, enable_lora=['q', 'k', 'v'], r=4, lora_alpha=8,
                        dropout_rate=0.1)
                    setattr(block, name, new_multi_head_lora)

    def encode_image(self, image, audio):
        return self.visual(image, audio)

    def encode_text(self, text, x):
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def encode_audio(self, x, mix_lambda):
        x = self.audio_encoder(x, mix_lambda)
        return x

    def forward(self, image, audio, label=None, mix_lambda=None):
        p_ctx = self.p_ctx
        n_ctx = self.n_ctx
        if p_ctx.dim() == 2:
            p_ctx = p_ctx.unsqueeze(0).expand(11, -1, -1)
            n_ctx = n_ctx.unsqueeze(0).expand(11, -1, -1)

        prefix = self.token_prefix
        p_suffix = self.p_token_suffix
        n_suffix = self.n_token_suffix

        p_prompts = torch.cat([prefix, p_ctx, p_suffix], dim=1)
        n_prompts = torch.cat([prefix, n_ctx, n_suffix], dim=1)

        x = torch.cat([p_prompts, n_prompts], dim=0)
        text = torch.cat([self.p_tokenized_prompts, self.n_tokenized_prompts]).to(x.device)

        audio_features = self.encode_audio(audio, mix_lambda)
        image_features = self.encode_image(image, audio_features)
        text_features = self.encode_text(text, x)

        # gate_in = torch.cat([image_features, audio_features], dim=-1)
        # gate_logits = self.gate(gate_in.detach())
        # w = F.log_softmax(gate_logits, dim=-1)
        # wv = w[:, 0].exp().unsqueeze(-1)
        # wa = w[:, 1].exp().unsqueeze(-1)

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        # audio_features = audio_features / audio_features.norm(dim=1, keepdim=True)

        p_features = text_features[:11, :]
        n_features = text_features[11:, :]

        logit_scale = self.logit_scale.clamp(max=math.log(100)).exp()
        image_p_logits = logit_scale * image_features @ p_features.t()
        image_n_logits = logit_scale * image_features @ n_features.t()
        logits = image_p_logits - image_n_logits

        # logit_scale_a = self.logit_scale_a.clamp(max=math.log(100)).exp()
        # audio_p_logits = logit_scale_a * audio_features @ p_features.t()
        # audio_n_logits = logit_scale_a * audio_features @ n_features.t()
        # audio_logits = audio_p_logits - audio_n_logits

        # logits = wv * image_logits + wa * audio_logits
        if label is not None:
            fused_ce = F.cross_entropy(logits, label)

            # img_ce = F.cross_entropy(image_logits, label, reduction='none')
            # aud_ce = F.cross_entropy(audio_logits, label, reduction='none')
            #
            # w_hat = torch.stack([-img_ce, -aud_ce], dim=-1)
            # w_hat = F.softmax(w_hat / 2, dim=-1)
            #
            # kl_loss = F.kl_div(w, w_hat.detach(), reduction='batchmean', log_target=False)
            #
            # loss = fused_ce + 0.5 * (((1 - wv.detach()).squeeze() * img_ce).mean() + (
            #             (1 - wa.detach()).squeeze() * aud_ce).mean()) + kl_loss

            neg_loss = F.relu((p_features * n_features).sum(dim=1)).mean()

            if self.training:
                return fused_ce, neg_loss
            else:
                return fused_ce, logits
        else:
            return logits
