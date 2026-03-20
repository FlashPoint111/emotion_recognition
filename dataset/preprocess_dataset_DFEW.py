import os
import random

from pyinstrument import Profiler
from tqdm import tqdm
import numpy as np
import pandas as pd
import yaml


def run_preprocessing(config, mode):
    dataset_dir = config["dataset"]["dir"]
    dataset_frames_dir = config["dataset"]['frames_dir']
    if mode == 'train':
        dataset_info_dir = config["train"]['label_dir']
    else:
        dataset_info_dir = config["test"]['label_dir']

    preprocessed_data = []

    info = pd.read_csv(dataset_info_dir)
    with tqdm(total=len(info)) as pbar:
        for video_info in info.itertuples():
            video_file, label = video_info[1], video_info[2]
            video_file = str(video_file)
            video_path = os.path.join(dataset_dir, video_file + '.flac')
            frame_path = os.path.join(dataset_frames_dir, str.zfill(video_file, 5))
            frame_num = len(os.listdir(frame_path))
            if frame_num < 3:
                continue
            label_idx = label

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
    with open("../config/AIM2.yaml", "r") as f:
        config = yaml.safe_load(f)
    preprocessed_data = run_preprocessing(config, mode="train")
