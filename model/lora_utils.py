from .clip_model import VisionTransformer
from .lora_layers import *

import torch
import torch.nn as nn
from typing import List
import clip
from clip.model import CLIP, build_model
from torch.cuda.amp import autocast, GradScaler
from .clip_model import VisionTransformer


def add_lora_to_clip_vision_encoder(
        clip_model,
        lora_r: int = 4,
        lora_alpha: int = 2,
        lora_dropout: float = 0.1,
        lora_parts: List[str] = ['q', 'k', 'v', 'o']
):
    vision_encoder = clip_model.transformer
    for block in vision_encoder.resblocks:
        original_mha = block.attn
        lora_mha = PlainMultiheadAttentionLoRA(
            original_mha,
            r=lora_r,
            lora_alpha=lora_alpha,
            dropout_rate=lora_dropout,
            enable_lora=lora_parts
        )
        lora_fc1 = LinearLoRA(block.mlp.c_fc, lora_r, lora_alpha, False, lora_dropout)
        lora_fc2 = LinearLoRA(block.mlp.c_proj, lora_r, lora_alpha, False, lora_dropout)
        block.attn = lora_mha
        block.mlp.c_fc = lora_fc1
        block.mlp.c_proj = lora_fc2

    print("Successfully added LoRA to the CLIP image encoder.")
    return clip_model


def prepare_lora_training(model: nn.Module, lr: float = 1e-4, weight_decay: float = 1e-2):
    for name, param in model.named_parameters():
        if 'temporal' in name:
            param.requires_grad = True
        elif 'lora_' not in name:
            param.requires_grad = False

    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    num_trainable = sum(p.numel() for p in trainable_params)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {num_trainable}")
    print(f"Total parameters: {total_params}")
    print(f"Trainable ratio: {100 * num_trainable / total_params:.2f}%")

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    return optimizer


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VisionTransformer(224, 16, 768, 12, 8, 768)
    model.init_weights("ViT-B/16")
    model = add_lora_to_clip_vision_encoder(
        model,
        lora_r=8,
        lora_alpha=32,
        lora_parts=['q', 'k', 'v', 'o']
    )
    model = model.to(device)
    optimizer = prepare_lora_training(model, lr=1e-4)

    dummy_images = torch.randn(4, 3, 16, 224, 224).to(device)
    dummy_labels = torch.randint(0, 1000, (4,)).to(device)
    loss_fn = nn.CrossEntropyLoss()

    print("\n--- Starting a simple training loop ---")
    model.train()
    scaler = GradScaler()
    for epoch in range(3):
        with autocast():
            image_features = model.forward(dummy_images)

        # 简化示例：假设有一个固定的分类头
        dummy_classifier_head = torch.randn(image_features.size(1), 1000).to(device)
        logits = image_features @ dummy_classifier_head

        loss = loss_fn(logits, dummy_labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")