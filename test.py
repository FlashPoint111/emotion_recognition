import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, accuracy_score, classification_report, recall_score
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset.preprocess_dataset_MAFW import *
from dataset.video_dataloader import ImageAudioDataset
from model.model import AIM, VideoClip, HTSAT, CLAIP
from dataset.data_preprocess import ResizeLongestSideAndPad
import yaml
from collections import OrderedDict
from collections import defaultdict
from train import _unwrap_model, _clean_state_dict_keys, validate
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy


if __name__ == '__main__':
    with open("./config/AIM.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda')
    cudnn.benchmark = True
    criterion = SoftTargetCrossEntropy()
    model = VideoClip(loss_fn=criterion).to(device)
    checkpoint_path = "./output/checkpoint_57.pth"
    if os.path.isfile(checkpoint_path):
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model_to_load = _unwrap_model(model)
        cleaned_state_dict = _clean_state_dict_keys(checkpoint['model'])
        msg = model_to_load.load_state_dict(cleaned_state_dict, strict=False)
        print(msg)
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(checkpoint_path, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(checkpoint_path))
    preprocess = None
    val_dataset_info = run_preprocessing(config, mode='val')
    val_dataset = ImageAudioDataset(config, val_dataset_info, preprocess, mode='val')
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=False)
    model.eval()
    data_loss, uar = validate(model, val_dataloader, device, 0)
    print(uar)
    '''all_labels = []
    all_preds = []
    result = {}

    for i, sample in enumerate(val_dataloader):
        frame = sample['frame']
        image = sample["video"].to(device, non_blocking=True)
        ori_audio = sample["audio"]
        input_dict = {}
        keys = ori_audio.keys()
        for k in keys:
            input_dict[k] = ori_audio[k].to(device, non_blocking=True)
        label = sample["label"].to(device, non_blocking=True)
        with torch.no_grad():
            logits = model(image)
        for j in range(len(frame)):
            temp = {'logits': [], 'label': label[j].cpu()}
            result.setdefault(frame[j], temp)['logits'].append(logits[j].cpu())

    for item in result:
        temp = result[item]
        logits = torch.stack(temp['logits']).mean(dim=0)
        logits = torch.softmax(logits, dim=0)
        all_preds.append(torch.argmax(logits))
        all_labels.append(temp['label'])

    all_preds = torch.stack(all_preds)
    all_labels = torch.stack(all_labels)

    emotions = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "contempt", "anxiety", "helplessness",
                "disappointment", "neutral"]
    # Overall accuracy
    accuracy = accuracy_score(all_labels.numpy(), all_preds.numpy())
    print("Overall Accuracy:", accuracy)

    # # Calculate per-class accuracy
    # class_accuracies = {}
    # unique_classes = torch.unique(all_labels)
    # for cls in unique_classes:
    #     cls = cls.item()
    #     cls_indices = (all_labels == cls)  # Mask for current class
    #     cls_correct = (all_preds[cls_indices] == all_labels[cls_indices]).sum().item()
    #     cls_total = cls_indices.sum().item()
    #     cls_accuracy = cls_correct / cls_total if cls_total > 0 else 0
    #     class_accuracies[cls] = cls_accuracy
    #
    # print("Per-Class Accuracy:")
    # for cls, acc in class_accuracies.items():
    #     print(f"{emotions[cls]}: {acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(all_labels.numpy(), all_preds.numpy()))

    uar = balanced_accuracy_score(all_labels.numpy(), all_preds.numpy())
    print(f"UAR (Unweighted Average Recall): {uar:.4f}")

    war = accuracy_score(all_labels.numpy(), all_preds.numpy())
    print(f"WAR (Weighted Average Recall): {war:.4f}")'''
