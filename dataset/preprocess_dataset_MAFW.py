import os
import random

from pyinstrument import Profiler
from tqdm import tqdm
import numpy as np


def run_preprocessing(config, mode):
    dataset_dir = config["dataset"]["dir"]
    dataset_frames_dir = config["dataset"]['frames_dir']
    if mode == 'train':
        dataset_info_dir = config["train"]['label_dir']
    else:
        dataset_info_dir = config["test"]['label_dir']

    emotions = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "contempt", "anxiety", "helplessness",
                "disappointment", "neutral"]
    preprocessed_data = []
    ban_list = ['00903.mp4', '03414.mp4', '03724.mp4', '03727.mp4', '03810.mp4', '03931.mp4', '03440.mp4', '03721.mp4',
                '03728.mp4', '03855.mp4', '03928.mp4', '03732.mp4', '03927.mp4', '09529.mp4', '03730.mp4', '03726.mp4']

    with open(dataset_info_dir, 'r') as f:
        info = f.readlines()
    with tqdm(total=len(info)) as pbar:
        for video_info in info:
            video_file, label = video_info.strip().split(' ')
            if video_file in ban_list:
                continue
            video_path = os.path.join(dataset_dir) #, video_file.replace('.mp4', '.wav'))
            frame_path = os.path.join(dataset_frames_dir, video_file.split('.')[0])
            with open(os.path.join(frame_path, "n_frames"), 'r') as f:
                frame_num = int(f.read())
                if frame_num < 3:
                    continue
            label_idx = emotions.index(label)

            video_info = {
                "video_path": video_path,
                "frame_path": frame_path,
                "frame_num": frame_num,
                "label": label_idx,
            }
            preprocessed_data.append(video_info)
            pbar.update(1)

    return preprocessed_data


if __name__ == '__main__':
    data = np.load("data_preprocess.npz")
    profiler = Profiler()
    profiler.start()
    config = {
        "dataset_dir": "F:/MAFW/data/clips",
        "label_dir": "F:/MAFW/Train & Test Set/single/no_caption/set_1/train.txt",
        "use_frame": True,
        "dataset_frames_dir": "F:/MAFW/data/frames",
        "required_frames": 16,
        "required_audio_len": 1,
        "sr": 16000,
    }
    preprocessed_data = run_preprocessing(config)
    profiler.stop()
    print(profiler.output_text(unicode=True, color=True))
