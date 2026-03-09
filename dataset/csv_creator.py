import json
import os

import cv2
import pandas as pd


def video_split(train_set, valid_set, video_path, image_path, subset, label_path):
    assert os.path.exists(image_path), f'{image_path} does not exist!'
    video_path = video_path + '.mp4' if os.path.exists(video_path + '.mp4') else video_path + '.avi'
    videoinfo = cv2.VideoCapture(video_path)
    if subset == 'Train_Set':
        train_set.loc[len(train_set)] = [video_path, image_path, label_path,
                                         videoinfo.get(cv2.CAP_PROP_FPS), videoinfo.get(cv2.CAP_PROP_FRAME_COUNT) + 1]
    else:
        valid_set.loc[len(valid_set)] = [video_path, image_path, label_path,
                                         videoinfo.get(cv2.CAP_PROP_FPS), videoinfo.get(cv2.CAP_PROP_FRAME_COUNT) + 1]


def main():
    with open('../config/data_path.json') as f:
        config = json.load(f)
    video_path = config['video_path']
    image_path = config['image_path']
    label_path = config['label_path']
    output_path = config['output_path']
    train_set = pd.DataFrame(data=None, columns=['video_path', 'image_path', 'label_path', 'fps', 'total_frame'])
    valid_set = pd.DataFrame(data=None, columns=['video_path', 'image_path', 'label_path', 'fps', 'total_frame'])
    for root, dirs, files in os.walk(label_path):
        for file in files:
            if '.DS_Store' in file:
                continue
            subset = 'Train_Set' if 'Train_Set' in root else 'Validation_Set'
            if 'left' in file or 'right' in file:
                video_split(train_set, valid_set, os.path.join(video_path, file.split('_')[0]),
                            os.path.join(image_path, file.split('.')[0]), subset,
                            os.path.join(root, file))
            else:
                video_split(train_set, valid_set, os.path.join(video_path, file.split('.')[0]),
                            os.path.join(image_path, file.split('.')[0]), subset,
                            os.path.join(root, file))

    train_set.to_csv(os.path.join(output_path, 'train.csv'), index=False, encoding='utf-8')
    valid_set.to_csv(os.path.join(output_path, 'valid.csv'), index=False, encoding='utf-8')


if __name__ == '__main__':
    main()
