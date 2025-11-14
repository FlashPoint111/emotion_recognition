import warnings
import argparse
import datetime
import json
import time
import os
import numpy as np
import open_clip
import yaml
import math
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report, recall_score
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import get_cosine_schedule_with_warmup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import utils.utils as utils
from dataset.preprocess_dataset_MAFW import *
from dataset.video_dataloader import ImageAudioDataset
from model.model import AIM, VideoClip, HTSAT, CLAIP
from torch import nn
from utils.randaugment import RandomAugment
from utils.mixup import Mixup
from functools import partial
from dataset.data_preprocess import ResizeLongestSideAndPad
from PIL import Image, ImageFile
from model.utils import get_mix_lambda, do_mixup
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import balanced_accuracy_score, classification_report
from collections import OrderedDict
warnings.filterwarnings("ignore")
ImageFile.LOAD_TRUNCATED_IMAGES = True


def _unwrap_model(model):
    """Return the underlying model, regardless of DDP/DataParallel wrappers."""
    return model.module if hasattr(model, "module") else model


def _clean_state_dict_keys(state_dict):
    """Remove a leading 'module.' prefix that can appear in multi-GPU checkpoints."""
    if not state_dict:
        return state_dict
    cleaned_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            cleaned_state_dict[k[7:]] = v
        else:
            cleaned_state_dict[k] = v
    return cleaned_state_dict


def create_optimizer_and_scheduler(model, total_steps, warmup_steps, last_epoch=-1):
    no_decay_param = ['audio_encoder.norm', 'temporal_embedding', 'temporal_cls', 'ln_post', 'temporal_norm', 'x_norm',
                      'a_norm']
    params_decay = []
    params_no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(p_name in name for p_name in no_decay_param) or name == 'b':
            params_no_decay.append(param)
        else:
            params_decay.append(param)
    optimizer_grouped_parameters = [
        {
            'params': params_decay,
            'weight_decay': 0.05
        },
        {
            'params': params_no_decay,
            'weight_decay': 0.0
        }
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=5e-5)
    for group in optimizer.param_groups:
        group['initial_lr'] = group['lr']
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        last_epoch=last_epoch
    )
    return optimizer, scheduler


def filter_grad_parameters(model):
    grad_params = {name: param for name, param in model.named_parameters() if param.requires_grad}
    return grad_params


def train(model, data_loader, optimizer, epoch, mixup_fn, device, scheduler, accumulation_steps):
    # Train
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = len(data_loader) // 4
    scaler = GradScaler(enabled=False)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    num_iters = len(data_loader)
    update_step = math.ceil(num_iters / accumulation_steps)
    now_step = 0
    H_v_total, H_a_total = [], []
    for i, sample in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        ori_image = sample["video"].to(device, non_blocking=True)
        ori_audio = sample["audio"]
        input_dict = {}
        keys = ori_audio.keys()
        for k in keys:
            input_dict[k] = ori_audio[k].to(device, non_blocking=True)
        ori_label = F.one_hot(sample["label"], num_classes=11).to(device, non_blocking=True)
        mix_lambda = torch.from_numpy(get_mix_lambda(0.5, ori_image.shape[0])).to(device)
        image = do_mixup(ori_image, mix_lambda)
        label = do_mixup(ori_label, mix_lambda)
        # image, label = mixup_fn(ori_image, ori_label)
        if now_step != update_step - 1:
            accum = accumulation_steps
        else:
            accum = num_iters - (update_step - 1) * accumulation_steps
        if epoch < 15:
            beta, gamma = 0.1, 1
        else:
            beta, gamma = 1, 0.1
        with autocast(dtype=torch.bfloat16):
            loss = model(image, label)
            loss = loss / accum
            '''loss, loss_v, loss_a, H_v, H_a = model(image, input_dict, label, mixup_lambda=mix_lambda)
            ori_loss = loss
            loss = beta * loss + gamma * (loss_v + loss_a)
            loss = loss / accum
        H_v_total.append(H_v.cpu().numpy())
        H_a_total.append(H_a.cpu().numpy())
        '''
        loss.backward()
        if (i != 0 and (i + 1) % accumulation_steps == 0) or ((i + 1) == num_iters) or accumulation_steps == 1:
            now_step += 1
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=scheduler.get_last_lr()[0])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    # print("H_v: {:.4f}, H_a: {:.4f}".format(np.array(H_v_total).mean(), np.array(H_a_total).mean()))
    return {k: "{:.6f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


def validate(model, data_loader, device, epoch):
    model.eval()
    data_loss = []
    pred = []
    label = []
    for i, sample in enumerate(data_loader):
        ori_image = sample["video"].to(device, non_blocking=True)
        ori_audio = sample["audio"]
        input_dict = {}
        keys = ori_audio.keys()
        for k in keys:
            input_dict[k] = ori_audio[k].to(device, non_blocking=True)
        ori_label = F.one_hot(sample["label"], num_classes=11).to(device, non_blocking=True)
        with torch.no_grad():
            #logits, loss, loss_v, loss_a, H_v, H_a = model(ori_image, input_dict, ori_label, mixup_lambda=None)
            logits, loss = model(ori_image, ori_label)
            data_loss.append(loss.item())
        logits_pred = torch.argmax(logits, dim=-1)
        label.extend(sample["label"].cpu().numpy())
        pred.extend(logits_pred.cpu().numpy())

    is_distributed = dist.is_available() and dist.is_initialized()
    if is_distributed:
        world_size = dist.get_world_size()
        gathered_preds = [None] * world_size
        gathered_labels = [None] * world_size
        gathered_losses = [None] * world_size
        dist.all_gather_object(gathered_preds, pred)
        dist.all_gather_object(gathered_labels, label)
        dist.all_gather_object(gathered_losses, data_loss)

        if utils.is_main_process():
            all_preds = [item for sublist in gathered_preds for item in sublist]
            all_labels = [item for sublist in gathered_labels for item in sublist]
            all_losses = [item for sublist in gathered_losses for item in sublist]
            uar = balanced_accuracy_score(np.array(all_labels), np.array(all_preds))
            print(classification_report(np.array(all_labels), np.array(all_preds)))
            data_loss = np.array(all_losses).mean()
            obj = [float(data_loss), float(uar)]
        else:
            obj = [0.0, 0.0]
        dist.broadcast_object_list(obj, src=0)
        data_loss, uar = obj[0], obj[1]
    else:
        uar = balanced_accuracy_score(np.array(label), np.array(pred))
        print(classification_report(np.array(label), np.array(pred)))
        data_loss = np.array(data_loss).mean()

    return data_loss, uar


def main():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_distributed = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if is_distributed:
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cudnn.benchmark = True
    with open("./config/AIM.yaml", "r") as f:
        config = yaml.safe_load(f)
    start_epoch = 0
    max_epoch = 100
    warmup_steps = 1e-5

    print("Creating dataset")
    preprocess = None

    dataset_info = run_preprocessing(config, mode='train')
    dataset = ImageAudioDataset(config, dataset_info, preprocess, mode='train')
    val_dataset_info = run_preprocessing(config, mode='val')
    val_dataset = ImageAudioDataset(config, val_dataset_info, preprocess, mode='val')
    batch_size = config["train"]["batch_size"]
    if is_distributed:
        train_sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler,
                                num_workers=4, pin_memory=True, drop_last=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, sampler=val_sampler,
                                    num_workers=4, pin_memory=True, drop_last=False)

    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                num_workers=4, pin_memory=True, drop_last=True)
        val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    print("Creating model")
    criterion = SoftTargetCrossEntropy()
    model = VideoClip(loss_fn=criterion).to(device)
    accumulation_steps = 4
    steps_per_epoch = math.ceil(len(dataloader) / accumulation_steps)
    total_steps = steps_per_epoch * max_epoch
    warmup_steps = steps_per_epoch * 5
    resume = config["train"]["resume"]
    checkpoint_path = config["train"]["checkpoint_path"]
    checkpoint = None
    optimizer_state_dict = None
    scheduler_state_dict = None

    if resume:
        if os.path.isfile(checkpoint_path):
            print("=> loading checkpoint '{}'".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model_to_load = _unwrap_model(model)
            cleaned_state_dict = _clean_state_dict_keys(checkpoint['model'])
            msg = model_to_load.load_state_dict(cleaned_state_dict, strict=False)
            print(msg)
            start_epoch = checkpoint['epoch'] + 1
            print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
            optimizer_state_dict = checkpoint.get('optimizer')
            scheduler_state_dict = checkpoint.get('lr_scheduler')
        else:
            print("=> no checkpoint found at '{}'".format(checkpoint_path))

    optimizer, scheduler = create_optimizer_and_scheduler(model, total_steps, warmup_steps, last_epoch=start_epoch - 1)
    if checkpoint is not None:
        if optimizer_state_dict is not None:
            msg = optimizer.load_state_dict(optimizer_state_dict)
            print(msg)
        if scheduler_state_dict is not None:
            msg = scheduler.load_state_dict(scheduler_state_dict)
            print(msg)
        del checkpoint

    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )

    output_dir = config["train"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    mixup_fn = Mixup(
        mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None,
        prob=1.0, switch_prob=0.5, mode='batch',
        label_smoothing=0.1, num_classes=11)

    print("Start training")
    start_time = time.time()
    best_loss = math.inf
    best_uar = 0

    for epoch in range(start_epoch, max_epoch):
        if is_distributed:
            dataloader.sampler.set_epoch(epoch)
        model.train()
        train_stats = train(model, dataloader, optimizer, epoch, mixup_fn, device, scheduler, accumulation_steps)
        model.eval()
        loss, uar = validate(model, val_dataloader, device, epoch)
        if utils.is_main_process():
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch}
            model_state = _unwrap_model(model).state_dict()
            save_obj = {
                'model': model_state,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': scheduler.state_dict(),
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(output_dir, 'checkpoint_%02d.pth' % epoch))
            with open(os.path.join(output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\t")
                f.write('VAL UAR:{:.4f}, VAL Loss:{:.4f}\n'.format(uar, loss))
            if uar > best_uar:
                best_uar = uar
                torch.save(save_obj, os.path.join(output_dir, 'best_checkpoint.pth'))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
