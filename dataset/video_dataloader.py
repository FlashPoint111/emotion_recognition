import torch
import torchaudio
from PIL import ImageFile
from torch.utils.data import Dataset
from torchvision import transforms
import math

ImageFile.LOAD_TRUNCATED_IMAGES = True
from .random_erasing import RandomErasing
from dataset.preprocess_dataset_MAFW import *
from dataset.preprocess_utils import *
from . import video_transforms
from . import volume_transforms
from dataset.audio_data import get_audio_features, int16_to_float32, float32_to_int16


def extract_audio(audio_path: str,
                  str_idx: int,
                  end_idx: int,
                  frame_idx: int,
                  sr=48000
                  ):
    # Read the audio file and create its log-Mel spectrum.
    audio_waveform, _ = librosa.load(audio_path, sr=sr)
    audio_waveform = int16_to_float32(float32_to_int16(audio_waveform))
    audio_waveform = torch.from_numpy(audio_waveform).float()
    length = len(audio_waveform)
    start = int(str_idx / (frame_idx - 1) * length)
    end = int(end_idx / (frame_idx - 1) * length)
    if start >= end:
        print(start, end)
    audio_waveform = audio_waveform[start:end]
    audio_cfg = {'audio_length': 1024, 'class_num': 527, 'clip_samples': 480000, 'fmax': 14000, 'fmin': 50,
                 'hop_size': 480, 'mel_bins': 64, 'model_name': 'base', 'model_type': 'HTSAT', 'sample_rate': 48000,
                 'window_size': 1024}
    temp_dict = {}
    temp_dict = get_audio_features(
        temp_dict, audio_waveform, 480000,
        data_truncating='rand_trunc',
        data_filling='repeatpad',
        audio_cfg=audio_cfg,
        require_grad=audio_waveform.requires_grad
    )
    return temp_dict


def extract_video(clip_path: str,
                  transform,
                  frame_num,
                  frames_per_segment,
                  segments: int = 8,
                  mode: str = 'train',
                  chunk_nb=None,
                  split_nb=None
                  ):
    sr = 4
    interval_length = frame_num // segments
    file_list = os.listdir(clip_path)
    file_list.sort(key=lambda x: int(x[:-4]))
    frame_list = []
    clip_num = 16
    clip_len = clip_num * sr

    if mode == 'train':
        if frame_num <= clip_len:
            index = np.linspace(0, frame_num, num=frame_num // sr)
            index = np.concatenate((index, np.ones(clip_num - frame_num // sr) * frame_num))
            index = np.clip(index, 0, frame_num - 1).astype(np.int64)
        else:
            end_idx = np.random.randint(clip_len, frame_num)
            str_idx = end_idx - clip_len
            index = np.linspace(str_idx, end_idx, num=clip_num)
            index = np.clip(index, str_idx, end_idx - 1).astype(np.int64)

    else:
        index = [x for x in range(0, frame_num, sr)]
        while len(index) < clip_num:
            index.append(index[-1])


    frame_list += [file_list[x] for x in index]
    video = []
    for idx in frame_list:
        frame_idx = os.path.join(clip_path, idx)
        try:
            with Image.open(frame_idx) as img:
                image = img.convert('RGB')
                video.append(image)
        except Exception as e:
            print(f"Error loading {frame_idx}: {e}")
    if mode == 'train':
        data_resize = video_transforms.Compose([
            video_transforms.Resize(size=(224, 224))
        ])
        video = data_resize(video)
        video = _aug_frame(video)
        return video, index[0], index[-1]
    elif mode == 'test':
        test_num_segment = 2
        data_resize = video_transforms.Compose([
            video_transforms.Resize(size=(224, 224), interpolation='bilinear')
        ])
        video = data_resize(video)
        if isinstance(video, list):
            video = np.stack(video, 0)
        #temporal_step = max(1.0 * (video.shape[0] - 16) / (test_num_segment - 1), 0)
        #temporal_start = int(chunk_nb * temporal_step)
        temporal_start = int(video.shape[0] / 2)
        video = video[temporal_start-8:temporal_start+8, ...]

        data_transform = video_transforms.Compose([
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                       std=[0.26862954, 0.26130258, 0.27577711])
        ])
        video = data_transform(video)
        return video, index[0], index[-1]
    elif mode == 'val':
        temporal_start = int(len(video) / 2)
        video = video[temporal_start - 8:temporal_start + 8]
        data_transform = video_transforms.Compose([
            video_transforms.Resize(224, interpolation='bilinear'),
            video_transforms.CenterCrop(size=(224, 224)),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                       std=[0.26862954, 0.26130258, 0.27577711])
        ])
        video = data_transform(video)
        return video, index[0], index[-1]


def _aug_frame(buffer):
    aug_transform = video_transforms.create_random_augment(
        input_size=(224, 224),
        auto_augment='rand-m7-n4-mstd0.5-inc1',
        interpolation='bicubic',
    )

    buffer = aug_transform(buffer)

    buffer = [transforms.ToTensor()(img) for img in buffer]
    buffer = torch.stack(buffer)  # T C H W
    buffer = buffer.permute(0, 2, 3, 1)  # T H W C

    buffer = tensor_normalize(
        buffer, [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]
    )
    # T H W C -> C T H W.
    buffer = buffer.permute(3, 0, 1, 2)

    scl, asp = (
        # me: org min scale is too small
        # [0.08, 1.0],
        [0.08, 1.0],
        [0.75, 1.3333],
    )

    buffer = spatial_sampling(
        buffer,
        spatial_idx=-1,
        min_scale=256,
        max_scale=320,
        crop_size=224,
        random_horizontal_flip=True,
        inverse_uniform_sampling=False,
        aspect_ratio=asp,
        scale=scl,
        motion_shift=False
    )

    erase_transform = RandomErasing(
        0.25,
        mode='pixel',
        max_count=1,
        num_splits=False,
        device="cpu",
    )
    buffer = buffer.permute(1, 0, 2, 3)
    buffer = erase_transform(buffer)
    buffer = buffer.permute(1, 0, 2, 3)

    return buffer


def spatial_sampling(
        frames,
        spatial_idx=-1,
        min_scale=256,
        max_scale=320,
        crop_size=224,
        random_horizontal_flip=True,
        inverse_uniform_sampling=False,
        aspect_ratio=None,
        scale=None,
        motion_shift=False,
):
    """
    Perform spatial sampling on the given video frames. If spatial_idx is
    -1, perform random scale, random crop, and random flip on the given
    frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
    with the given spatial_idx.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `num frames` x `height` x `width` x `channel`.
        spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
            or 2, perform left, center, right crop if width is larger than
            height, and perform top, center, buttom crop if height is larger
            than width.
        min_scale (int): the minimal size of scaling.
        max_scale (int): the maximal size of scaling.
        crop_size (int): the size of height and width used to crop the
            frames.
        inverse_uniform_sampling (bool): if True, sample uniformly in
            [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
            scale. If False, take a uniform sample from [min_scale,
            max_scale].
        aspect_ratio (list): Aspect ratio range for resizing.
        scale (list): Scale range for resizing.
        motion_shift (bool): Whether to apply motion shift for resizing.
    Returns:
        frames (tensor): spatially sampled frames.
    """
    assert spatial_idx in [-1, 0, 1, 2]
    if spatial_idx == -1:
        if aspect_ratio is None and scale is None:
            frames, _ = video_transforms.random_short_side_scale_jitter(
                images=frames,
                min_size=min_scale,
                max_size=max_scale,
                inverse_uniform_sampling=inverse_uniform_sampling,
            )
            frames, _ = video_transforms.random_crop(frames, crop_size)
        else:
            transform_func = (
                video_transforms.random_resized_crop_with_shift
                if motion_shift
                else video_transforms.random_resized_crop
            )
            frames = transform_func(
                images=frames,
                target_height=crop_size,
                target_width=crop_size,
                scale=scale,
                ratio=aspect_ratio,
            )
        if random_horizontal_flip:
            frames, _ = video_transforms.horizontal_flip(0.5, frames)
    else:
        # The testing is deterministic and no jitter should be performed.
        # min_scale, max_scale, and crop_size are expect to be the same.
        assert len({min_scale, max_scale, crop_size}) == 1
        frames, _ = video_transforms.random_short_side_scale_jitter(
            frames, min_scale, max_scale
        )
        frames, _ = video_transforms.uniform_crop(frames, crop_size, spatial_idx)
    return frames


def tensor_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor


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
        if self.mode == 'test':
            self.test_num_segment = 2
            self.test_seg = []
            self.test_dataset = []
            self.test_clip_dataset = []
            self.test_label_array = []
            self.test_dataset_frame_num = []
            for ck in range(self.test_num_segment):
                for idx in range(len(self.dataset_info)):
                    sample = self.dataset_info[idx]
                    self.test_label_array.append(sample['label'])
                    self.test_dataset.append(sample['frame_path'])
                    self.test_clip_dataset.append(sample['video_path'])
                    self.test_dataset_frame_num.append(sample['frame_num'])
                    self.test_seg.append(ck)
            print(len(self.test_seg))

    def __getitem__(self, idx):
        if self.mode == 'train' or self.mode == 'val':
            sample = self.dataset_info[idx]
            video_path = sample['video_path']
            frame_path = sample['frame_path']
            frame_num = sample['frame_num']
            label = sample['label']
            video, str_idx, end_idx = extract_video(frame_path, self.image_transform, frame_num, self.frames_per_segment,
                                                    self.segments,
                                                    self.mode)
            audio = extract_audio(video_path, str_idx, end_idx, frame_num)
            return {'video': video, 'audio': audio, 'label': label}
        else:
            video_path = self.test_clip_dataset[idx]
            frame_path = self.test_dataset[idx]
            label = self.test_label_array[idx]
            frame_num = self.test_dataset_frame_num[idx]
            chunk_nb = self.test_seg[idx]
            video, str_idx, end_idx = extract_video(frame_path, self.image_transform, frame_num, self.frames_per_segment, self.segments,
                                  self.mode, chunk_nb)
            audio = extract_audio(video_path, str_idx, end_idx, frame_num)
            return {'frame': frame_path.split('\\')[-1], 'video': video, 'audio': audio, 'label': label}

    def __len__(self):
        if self.mode == 'train' or self.mode == 'val':
            return len(self.dataset_info)
        else:
            return len(self.test_dataset)


def img_process(img: Image, transform: transforms.Compose):
    return transform(img)
