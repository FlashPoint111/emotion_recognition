import os

import ffmpeg
import librosa
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

def extract_frames(image_root: str, total_frame: int):
    image = []
    for file in range(0, total_frame):
        image_path = os.path.join(image_root, str(file+1).zfill(5)+'.jpg')
        try:
            image.append(np.array(Image.open(image_path).convert('RGB')).astype(np.float32) / 255.0)
        except FileNotFoundError:
            image.append(None)
    return image


def extract_audio(video_path: str, sr: int = 16000):
    # Read the audio file and create its log-Mel spectrum.
    audio, _ = (
        ffmpeg.input(video_path)
        .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
        .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True, quiet=True)
    )
    audio = np.frombuffer(audio, np.int16).flatten().astype(np.float32) / 32768.0
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=400, hop_length=160, n_mels=128)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    return log_mel_spectrogram


def extract_label(label_path: str):
    output = []
    with open(label_path, "r") as f:
        for line in f.readlines()[1:]:
            if '-5' in line:
                output.append(None)
            else:
                output.append(np.array(line.strip('\n').split(','), dtype=np.float32))
        return output


class ImageAudioDataset(Dataset):
    def __init__(self, configs, image_transform):
        self.image_transform = image_transform
        self.dataset_dir = configs["dataset_path"]
        self.frame_dir = configs["frame_path"] if configs["use_frame"] else None
        self.data_idx = np.random.permutation(self.max_len)[:1000]

    def __getitem__(self, idx):
        try:
            sample = torch.load(os.path.join(self.data_dir, self.data_list[self.data_idx[idx]]))
        except FileNotFoundError:
            raise FileNotFoundError
        sample["video"] = list(map(lambda x: img_process(x, self.image_transform), sample["video"]))
        sample["video"] = torch.stack(sample["video"], dim=0).permute(1, 0, 2, 3)
        sample["label"] = np.array(sample["label"], dtype=np.float32)
        sample["audio"] = torch.tensor(sample["audio"], dtype=torch.float32)
        return sample

    def __len__(self):
        return self.max_len