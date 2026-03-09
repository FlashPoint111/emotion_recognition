import torch
import torch.backends.cudnn as cudnn
import yaml
from sklearn.metrics import balanced_accuracy_score, accuracy_score, classification_report, confusion_matrix
from timm.loss import SoftTargetCrossEntropy
from torch.utils.data import DataLoader

from dataset.preprocess_dataset_MAFW import *
from dataset.video_dataloader import ImageAudioDataset
from model.model import CLAIP

if __name__ == '__main__':
    with open("./config/AIM.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda')
    cudnn.benchmark = True
    criterion = None #SoftTargetCrossEntropy()
    model = CLAIP(loss_fn=criterion).to(device)
    checkpoint_path = "./output/best_checkpoint.pth"
    if os.path.isfile(checkpoint_path):
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        # model_to_load = _unwrap_model(model)
        # cleaned_state_dict = _clean_state_dict_keys(checkpoint['model'])
        # msg = model_to_load.load_state_dict(cleaned_state_dict, strict=False)
        msg = model.load_state_dict(checkpoint['model'], strict=False)
        print(msg)
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(checkpoint_path, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(checkpoint_path))
    model.eval()
    preprocess = None
    val_dataset_info = run_preprocessing(config, mode='val')
    val_dataset = ImageAudioDataset(config, val_dataset_info, preprocess, mode='test')
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=False)
    all_labels = []
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
        with (torch.no_grad()):
            logits, image_logits, audio_logits = model(image, input_dict)
        for j in range(len(frame)):
            temp = {'logits': [], 'label': label[j].cpu(), 'image_logits': [], 'audio_logits': []}
            result.setdefault(frame[j], temp)['logits'].append(logits[j].cpu())
            result.setdefault(frame[j], temp)['image_logits'].append(image_logits[j].cpu())
            result.setdefault(frame[j], temp)['audio_logits'].append(audio_logits[j].cpu())

    all_r = []
    total_agree = 0
    agree_pred = []
    agree_label = []
    disagree_pred = []
    disagree_label = []
    disagree_pred_image = []
    agree_pred_image = []

    for item in result:
        temp = result[item]
        logits = torch.stack(temp['logits']).mean(dim=0)
        image_logits = torch.stack(temp['image_logits']).mean(dim=0)
        audio_logits = torch.stack(temp['audio_logits']).mean(dim=0)
        logits = torch.softmax(logits, dim=0)
        all_preds.append(torch.argmax(logits))
        all_labels.append(temp['label'])
        r = audio_logits.norm() / image_logits.norm()
        all_r.append(r)
        agree = (logits.argmax() == image_logits.argmax())
        total_agree = total_agree + agree
        if agree:
            agree_pred.append(torch.argmax(logits))
            agree_label.append(temp['label'])
            agree_pred_image.append(torch.argmax(image_logits))
        else:
            disagree_pred.append(torch.argmax(logits))
            disagree_label.append(temp['label'])
            disagree_pred_image.append(torch.argmax(image_logits))

    all_preds = torch.stack(all_preds)
    all_labels = torch.stack(all_labels)
    agree_pred = torch.stack(agree_pred)
    agree_label = torch.stack(agree_label)
    disagree_pred = torch.stack(disagree_pred)
    disagree_label = torch.stack(disagree_label)
    disagree_pred_image = torch.stack(disagree_pred_image)
    agree_pred_image = torch.stack(agree_pred_image)

    all_r = torch.stack(all_r)
    print(f"The norm ratio is {all_r.mean()}")

    emotions = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise", "contempt", "anxiety",
                      "helplessness",
                      "disappointment"]
    # Overall accuracy
    accuracy = accuracy_score(all_labels.numpy(), all_preds.numpy())
    print("Overall Accuracy:", accuracy)

    print("\nClassification Report:")
    print(classification_report(all_labels.numpy(), all_preds.numpy()))

    uar = balanced_accuracy_score(all_labels.numpy(), all_preds.numpy())
    print(f"UAR (Unweighted Average Recall): {uar:.4f}")

    war = accuracy_score(all_labels.numpy(), all_preds.numpy())
    print(f"WAR (Weighted Average Recall): {war:.4f}")

    print(f"Non Flip ratio={total_agree / len(result):.4f}")

    print(f"\nNon Flip group:\n {classification_report(agree_label.numpy(), agree_pred.numpy())}")

    print(f"\nNon Flip group image:\n {classification_report(agree_label.numpy(), agree_pred_image.numpy())}")

    print(f"\nFlip group:\n {confusion_matrix(disagree_label.numpy(), disagree_pred.numpy())}")

    print(f"\nFlip group image:\n {confusion_matrix(disagree_label.numpy(), disagree_pred_image.numpy())}")

