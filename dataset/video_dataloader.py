import torch
import torchaudio
from PIL import ImageFile
from torch.utils.data import Dataset
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True
from .random_erasing import RandomErasing
from dataset.preprocess_dataset_MAFW import *
from dataset.preprocess_utils import *
from . import video_transforms
from . import volume_transforms


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
                  chunk_nb=None,
                  split_nb=None
                  ):
    interval_length = frame_num // segments
    frames_idx = []
    if mode == 'train':
        converted_len = 16
        seg_len = frame_num

        if seg_len <= converted_len:
            frames_idx = np.arange(seg_len)
            frames_idx = np.concatenate((frames_idx, np.ones(16 - seg_len) * (seg_len - 1)))
        else:
            end_idx = np.random.randint(converted_len, seg_len)
            str_idx = end_idx - converted_len
            frames_idx = np.arange(str_idx, end_idx)

    else:
        frames_idx = np.arange(frame_num)

    video = []
    for idx in frames_idx:
        frame_idx = os.path.join(clip_path, str(idx).zfill(4) + ".png")
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
        return video
    else:
        test_num_crop = 2
        test_num_segment = 2
        data_resize = video_transforms.Compose([
            video_transforms.Resize(size=(224, 224), interpolation='bilinear')
        ])
        video = data_resize(video)
        if isinstance(video, list):
            video = np.stack(video, 0)
        spatial_step = 1.0 * (max(video.shape[1], video.shape[2]) - 224) \
                       / (test_num_crop - 1)
        temporal_step = max(1.0 * (video.shape[0] - 16) \
                            / (test_num_segment - 1), 0)
        temporal_start = int(chunk_nb * temporal_step)
        spatial_start = int(split_nb * spatial_step)
        if video.shape[1] >= video.shape[2]:
            video = video[temporal_start:temporal_start + 16, \
                    spatial_start:spatial_start + 224, :, :]
        else:
            video = video[temporal_start:temporal_start + 16, \
                    :, spatial_start:spatial_start + 224, :]

        data_transform = video_transforms.Compose([
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                           std=[0.26862954, 0.26130258, 0.27577711])
            ])
        video = data_transform(video)
        return video


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
        if self.mode != 'train':
            self.test_num_segment = 1
            self.test_num_crop = 1
            self.test_seg = []
            self.test_dataset = []
            self.test_label_array = []
            self.test_dataset_frame_num = []
            for ck in range(self.test_num_segment):
                for cp in range(self.test_num_crop):
                    for idx in range(len(self.dataset_info)):
                        sample = self.dataset_info[idx]
                        self.test_label_array.append(sample['label'])
                        self.test_dataset.append(sample['frame_path'])
                        self.test_dataset_frame_num.append(sample['frame_num'])
                        self.test_seg.append((ck, cp))

    def __getitem__(self, idx):
        if self.mode == 'train':
            sample = self.dataset_info[idx]
            video_path = sample['video_path']
            frame_path = sample['frame_path']
            frame_num = sample['frame_num']
            label = sample['label']
            video = extract_video(frame_path, self.image_transform, frame_num, self.frames_per_segment, self.segments,
                                  self.mode)
            return {'video': video, 'label': label}
        else:
            frame_path = self.test_dataset[idx]
            label = self.test_label_array[idx]
            frame_num = self.test_dataset_frame_num[idx]
            chunk_nb, split_nb = self.test_seg[idx]
            video = extract_video(frame_path, self.image_transform, frame_num, self.frames_per_segment, self.segments,
                                  self.mode, chunk_nb, split_nb)
            return {'frame': frame_path.split('\\')[-1], 'video': video, 'label': label}

    def __len__(self):
        if self.mode == 'train':
            return len(self.dataset_info)
        else:
            return len(self.test_dataset)


def img_process(img: Image, transform: transforms.Compose):
    return transform(img)
