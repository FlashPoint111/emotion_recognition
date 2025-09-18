import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from dataset.preprocess_dataset_MAFW import *
from dataset.preprocess_utils import *
from utils.randaugment import RandomAugment


def extract_audio(audio_path: str,
                  mean: float,
                  std: float,
                  mode: str,
                  segments: int = 8,
                  ):
    # Read the audio file and create its log-Mel spectrum.
    # audio, _ = (
    #     ffmpeg
    #     .input(audio_path)
    #     .output('pipe:', format='wav', ac=1, ar=16000)
    #     .run(capture_stdout=True, capture_stderr=True)
    # )
    waveform, sr = torchaudio.load(audio_path)
    waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform - waveform.mean()
    mel_spectrogram = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr,
                                                        use_energy=False, window_type='hanning',
                                                        num_mel_bins=128,
                                                        dither=0.0, frame_shift=10).transpose(1, 0)
    mel_spectrogram = (mel_spectrogram - mean) / (std * 2)
    audio_len = mel_spectrogram.shape[1]
    audio_seg_points = np.round(np.linspace(0, audio_len - 1, segments + 1)).astype(int)
    output = []
    # freqm = torchaudio.transforms.FrequencyMasking(6)
    # timem = torchaudio.transforms.TimeMasking(3)
    for i in range(segments):
        interval = audio_seg_points[i + 1] - audio_seg_points[i]
        if interval <= 32:
            p = 32 - interval
            m = torch.nn.ZeroPad2d((0, p, 0, 0))
            block = m(mel_spectrogram[:, audio_seg_points[i]: audio_seg_points[i + 1]])
        else:
            if mode == 'train':
                start_point = audio_seg_points[i] + np.random.randint(0, interval - 32)
                block = mel_spectrogram[:, start_point:start_point + 32]
            else:
                start_point = int((audio_seg_points[i] + audio_seg_points[i + 1]) / 2)
                block = mel_spectrogram[:, start_point - 16:start_point + 16]
        # if mode == 'train':
        #     block = freqm(block)
        #     block = timem(block)
        output.append(block)
    return torch.stack(output, dim=0)


def extract_video(clip_path: str,
                  transform,
                  frame_num,
                  frames_per_segment,
                  segments: int = 8,
                  mode: str = 'train',
                  ):
    interval_length = frame_num // segments
    frames_idx = []
    if mode == 'train':
        for i in range(segments):
            interval_start = i * interval_length
            interval_end = interval_start + interval_length if interval_start + interval_length < frame_num else frame_num
            frames_idx.extend(
                np.random.choice(np.arange(interval_start, interval_end), size=frames_per_segment, replace=False))
        frames_idx = sorted(frames_idx)
    else:
        frames_idx = np.linspace(0, frame_num, int(segments * frames_per_segment + 2), dtype=int)
        frames_idx = frames_idx[1:-1]
    video = []
    for idx in frames_idx:
        frame_idx = os.path.join(clip_path, str(idx).zfill(4) + ".png")
        try:
            with Image.open(frame_idx) as img:
                image = img.convert('RGB')
                video.append(image)
        except Exception as e:
            print(f"Error loading {frame_idx}: {e}")

    video = list(map(lambda x: img_process(x, transform), video))
    video = torch.stack(video, dim=0).permute(1, 0, 2, 3)

    return video


class ImageAudioDataset(Dataset):
    def __init__(self, config, dataset_info, image_transform, mode='train'):
        self.image_transform = image_transform
        self.dataset_info = dataset_info
        self.sr = config["dataset"]['sr']
        # self.required_frames = config["dataset"]['n_frame']
        self.mode = mode
        self.segments = config["dataset"]['n_frame']
        self.frames_per_segment = 1
        self.audio_mean = -8.4043
        self.audio_std = 4.6940

    def __getitem__(self, idx):
        sample = self.dataset_info[idx]
        video_path = sample['video_path']
        frame_path = sample['frame_path']
        frame_num = sample['frame_num']
        label = sample['label']
        video = extract_video(frame_path, self.image_transform, frame_num, self.frames_per_segment, self.segments,
                              self.mode)
        return {'video': video, 'label': label}

    def __len__(self):
        return len(self.dataset_info)


def img_process(img: Image, transform: transforms.Compose):
    return transform(img)


if __name__ == '__main__':

    normalize = transforms.Normalize((0.3527, 0.2792, 0.2490), (0.2430, 0.2075, 0.1999))
    pretrain_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        transforms.ToTensor(),
        normalize,
    ])

    config = {
        "dataset_dir": "F:/MAFW/data/clips",
        "label_dir": "F:/MAFW/Train & Test Set/single/no_caption/set_1/test.txt",
        "use_frame": True,
        "dataset_frames_dir": "F:/MAFW/data/frames",
        "required_frames": 16,
        "required_audio_len": 1,
        "sr": 16000,
        'segments': 8,
    }
    dataset_info = run_preprocessing(config)
    dataset = ImageAudioDataset(config, dataset_info, pretrain_transform, mode='val')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for data in dataloader:
        print(len(data))
