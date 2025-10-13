import json
import os

import ffmpeg
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
import librosa
import torch


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


if __name__ == '__main__':
    # Read config and choose video with fps=30
    with open('../config/train_dataset.json') as f:
        config = json.load(f)
    dataset = pd.read_csv(config["dataset_path"])
    dataset = dataset[dataset.fps == 30].to_dict(orient='records')

    transform = transforms.Compose([transforms.ToTensor()])
    sample_rate = config["sample_rate"]
    seq_len = config["seq_len"]
    
    #stride = config["stride"]
    if not os.path.exists(config["output_path"]):
        os.makedirs(config["output_path"])
    output_count = 0
    preprocess = np.load('data_preprocess.npz')
    image_mean, image_std = preprocess['total_image_mean'], preprocess['total_image_std']
    audio_mean, audio_std = preprocess['total_audio_mean'], preprocess['total_audio_std']

    audio_seq_len = int(seq_len / 30 * 100)

    for sample in dataset:
        print(sample['video_path'])
        label = extract_label(sample['label_path'])
        total_frame = len(label)
        image = extract_frames(sample['image_path'], total_frame)
        audio = extract_audio(sample['video_path'], sample_rate)

        i = 0
        j = 0
        audio_len = audio.shape[1]
        while i < audio_len:
            # Create an audio sample, if the remaining part is less than audio_seq_len then pad 0.
            if i + audio_seq_len > audio_len:
                audio_sample = np.zeros((audio.shape[0], audio_seq_len), dtype=np.float32)
                audio_sample[:, 0:audio_len-i] = ((audio[:, i:].T - audio_mean) / audio_std).T
            else:
                audio_sample = ((audio[:, i:i+audio_seq_len].T - audio_mean) / audio_std).T

            # Create a video sample, if a frame doesn't contain a face then pad None into the image_sample and label_sample.
            # If the remaining part is less than seq_len then pad None into these two lists.
            label_sample = []
            image_sample = []
            active = 0
            for idx in range(j, j+30):
                if idx >= total_frame:
                    label_sample.append(np.array([np.nan, np.nan]))
                    image_sample.append(np.zeros((112, 112, 3), dtype=np.float32))
                elif label[idx] is None or image[idx] is None:
                    label_sample.append(np.array([np.nan, np.nan]))
                    image_sample.append(np.zeros((112, 112, 3), dtype=np.float32))
                else:
                    label_sample.append(label[idx])
                    image_sample.append((image[idx] - image_mean) / image_std)
                    active += 1
            if active / seq_len >= 0.8:
                torch.save(({"video": image_sample, "audio": audio_sample, "label": label_sample}),
                        os.path.join(config["output_path"], f"{output_count}.pth"))
                output_count += 1
            i += audio_seq_len
            j += seq_len

