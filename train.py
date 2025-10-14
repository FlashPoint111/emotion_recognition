import argparse
import datetime
import json
import time
import clip
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
from model.model import AIM, VideoClip
from utils.randaugment import RandomAugment
from utils.mixup import Mixup
from functools import partial
from dataset.data_preprocess import ResizeLongestSideAndPad
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def create_optimizer_and_scheduler(model, total_steps, warmup_steps):
    no_decay_param = ['class_embedding', 'temporal_embedding', 'temporal_cls']
    params_decay = []
    params_no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(p_name in name for p_name in no_decay_param):
            params_no_decay.append(param)
        else:
            params_decay.append(param)
    optimizer_grouped_parameters = [
        {
            'params': params_decay,
            'weight_decay': 0.02
        },
        {
            'params': params_no_decay,
        }
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=1e-5)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    return optimizer, scheduler


def filter_grad_parameters(model):
    grad_params = {name: param for name, param in model.named_parameters() if param.requires_grad}
    return grad_params


def train(model, data_loader, optimizer, epoch, mixup_fn, device, scheduler):
    # Train
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = len(data_loader) // 4
    scaler = GradScaler()
    model.train()
    accumulation_steps = 4
    num_iters = len(data_loader)
    for i, sample in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        ori_image = sample["video"].to(device, non_blocking=True)
        ori_label = sample["label"].to(device, non_blocking=True)
        image, label = mixup_fn(ori_image, ori_label)

        remain = num_iters - i
        cur_accum = accumulation_steps if remain > accumulation_steps else remain
        with autocast():
            loss = model(image, label)
            loss = loss / cur_accum

        scaler.scale(loss).backward()
        if ((i + 1) % accumulation_steps == 0) or ((i + 1) == num_iters):
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=scheduler.get_last_lr()[0])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.6f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


def eval(model, data_loader, device):
    model.eval()
    all_labels = []
    all_preds = []

    for i, sample in enumerate(data_loader):
        image = sample["video"].to(device, non_blocking=True)
        label = sample["label"].to(device, non_blocking=True)
        with torch.no_grad():
            logits = model(image)
            predicted_classes = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
        all_preds.append(predicted_classes.cpu())
        all_labels.append(label.cpu())
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    uar = recall_score(all_labels.numpy(), all_preds.numpy(), average='macro')
    print("UAR:{:.4f}".format(uar))
    return all_labels, all_preds


def main():
    device = torch.device('cuda')
    cudnn.benchmark = True
    with open("./config/AIM.yaml", "r") as f:
        config = yaml.safe_load(f)
    start_epoch = 0
    max_epoch = 100
    warmup_steps = 1e-5

    print("Creating dataset")
    _, preprocess = clip.load("ViT-B/16", device=device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    pretrain_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    dataset_info = run_preprocessing(config, mode='train')
    dataset = ImageAudioDataset(config, dataset_info, preprocess)
    batch_size = config["train"]["batch_size"]
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    val_dataset_info = run_preprocessing(config, mode='val')
    val_dataset = ImageAudioDataset(config, val_dataset_info, preprocess, mode='val')
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=False)

    print("Creating model")
    # model = AIM(config)
    criterion = SoftTargetCrossEntropy()
    model = VideoClip(loss_fn=criterion)
    model = model.to(device)

    steps_per_epoch = math.ceil(len(dataloader) / 4)
    total_steps = steps_per_epoch * max_epoch
    warmup_steps = steps_per_epoch * 5

    optimizer, scheduler = create_optimizer_and_scheduler(model, total_steps, warmup_steps)

    resume = config["train"]["resume"]
    checkpoint_path = config["train"]["checkpoint_path"]
    if resume:
        if os.path.isfile(checkpoint_path):
            print("=> loading checkpoint '{}'".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            msg = model.load_state_dict(checkpoint['model'], strict=False)
            print(msg)
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch'] + 1
            print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(checkpoint_path))

    output_dir = config["train"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    mixup_fn = Mixup(
        mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None,
        prob=1.0, switch_prob=0.5, mode='batch',
        label_smoothing=0.1, num_classes=11)


    print("Start training")
    start_time = time.time()

    best_score = 0
    early_stop = 0
    best_epoch = 0
    for epoch in range(start_epoch, max_epoch):
        # if early_stop >= 5:
        #     break
        train_stats = train(model, dataloader, optimizer, epoch, mixup_fn, device, scheduler)
        if utils.is_main_process():
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch}
            save_obj = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': scheduler.state_dict(),
                'epoch': epoch,
            }
            # if epoch > 25:
            #     label, prediction = eval(model, val_dataloader, device)
            # print(classification_report(label.numpy(), prediction.numpy()))
            torch.save(save_obj, os.path.join(output_dir, 'checkpoint_%02d.pth' % epoch))
            # if score > best_score:
            #     best_score = score
            #     early_stop = 0
            #    best_epoch = epoch
            # else:
            #     early_stop += 1
            with open(os.path.join(output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")
    # label, prediction = eval(model, val_dataloader, device)
    # print(classification_report(label.numpy(), prediction.numpy()))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    # print('Best UAR epoch:{}, UAR={}'.format(best_epoch, best_score))


if __name__ == '__main__':
    main()
