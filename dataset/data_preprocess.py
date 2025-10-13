import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from dataset.preprocess_dataset_MAFW import *
from dataset.preprocess_utils import *


class ImageAudioDataset(Dataset):
    def __init__(self, config, dataset_info):
        self.image_transform = transforms.Compose([
            ResizeLongestSideAndPad(target_size=224, fill_value=0),
            transforms.ToTensor(), ])
        self.dataset_info = dataset_info

    def __getitem__(self, idx):
        sample = self.dataset_info[idx]
        video_path = sample['video_path']
        frame_path = sample['frame_path']
        frame_num = sample['frame_num']

        video = []
        for idx in range(frame_num):
            frame_idx = os.path.join(frame_path, str(idx).zfill(4) + ".png")
            video.append(Image.open(frame_idx).convert('RGB'))
        try:
            video = list(map(lambda x: img_process(x, self.image_transform), video))
            video = torch.stack(video, dim=0).permute(1, 0, 2, 3)
            waveform, sr = torchaudio.load(video_path)
            waveform = waveform.mean(dim=0, keepdim=True)
            waveform = waveform - waveform.mean()
            mel_spectrogram = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr,
                                                                use_energy=False, window_type='hanning',
                                                                num_mel_bins=128,
                                                                dither=0.0, frame_shift=10).transpose(1, 0)
            return video, mel_spectrogram
        except:
            print(video_path)

    def __len__(self):
        return len(self.dataset_info)


class ResizeLongestSideAndPad:
    def __init__(self, target_size=224, fill_value=0):
        self.target_size = target_size
        self.fill_value = fill_value

    def __call__(self, img):
        w, h = img.size
        if w == h == self.target_size:
            return img
        elif h > w:
            scale = self.target_size / float(h)
        else:
            scale = self.target_size / float(w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.BILINEAR)
        pad_w = self.target_size - new_w
        pad_h = self.target_size - new_h
        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top
        img = transforms.functional.pad(img, (left, top, right, bottom), fill=self.fill_value)
        return img


def img_process(img: Image, transform: transforms.Compose):
    return transform(img)


if __name__ == "__main__":
    config = {
        "dataset_dir": "F:/MAFW/data/clips",
        "label_dir": "F:/MAFW/Train & Test Set/total.txt",
        "use_frame": True,
        "dataset_frames_dir": "F:/MAFW/data/frames",
        "required_frames": 16,
        "required_audio_len": 1,
        "sr": 16000,
        'segments': 8,
    }
    dataset_info = run_preprocessing(config)
    dataset = ImageAudioDataset(config, dataset_info)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=8, drop_last=True)
    total_video_num = 0
    total_audio_num = 0
    total_pixel_mean = torch.zeros(3)
    total_pixel_var = torch.zeros(3)
    total_audio_mean = 0
    total_audio_var = 0

    for video, mel_spectrogram in dataloader:
        video = video.squeeze()
        mel_spectrogram = mel_spectrogram.squeeze()
        video_len = video.shape[1]
        audio_len = mel_spectrogram.shape[1]
        video_mean = torch.mean(video, dim=(1, 2, 3))
        video_var = torch.var(video, dim=(1, 2, 3))
        audio_mean = torch.mean(mel_spectrogram)
        audio_var = torch.var(mel_spectrogram)
        total_pixel_mean_tmp = (total_pixel_mean * total_video_num + video_mean * video_len) / (
                    total_video_num + video_len)
        total_audio_mean_tmp = (total_audio_mean * total_audio_num + audio_mean * audio_len) / (
                    total_audio_num + audio_len)
        total_pixel_var = ((total_pixel_var + torch.pow((total_pixel_mean - total_pixel_mean_tmp),
                                                        2)) * total_video_num +
                           (video_var + torch.pow((video_mean - total_pixel_mean_tmp), 2)) * video_len) / (
                                      total_video_num + video_len)
        total_audio_var = ((total_audio_var + torch.pow((total_audio_mean - total_audio_mean_tmp),
                                                        2)) * total_audio_num +
                           (audio_var + torch.pow((audio_mean - total_audio_mean_tmp), 2)) * audio_len) / (
                                      total_audio_num + audio_len)
        total_pixel_mean = total_pixel_mean_tmp
        total_audio_mean = total_audio_mean_tmp
        total_video_num = total_video_num + video_len
        total_audio_num = total_audio_num + audio_len
    print(total_pixel_mean, torch.sqrt(total_pixel_var))
    print(total_audio_mean, torch.sqrt(total_audio_var))
