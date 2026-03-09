import os


def run_preprocessing(config):
    dataset_dir = config['dataset_dir']
    dataset_info_dir = config['label_dir']
    emotions = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "contempt", "anxiety", "helplessness", "disappointment", "neutral"]
    preprocessed_data = []

    with open(dataset_info_dir, 'r') as f:
        info = f.readlines()

    for video_info in info:
        video_file, label = video_info.strip().split(' ')
        video_path = os.path.join(dataset_dir, video_file)
        label_idx = emotions.index(label)

        video_info = {
            "video_path": video_path,
            "label": label_idx
        }
        preprocessed_data.append(video_info)

    return preprocessed_data


if __name__ == '__main__':
    config = {
        "dataset_dir": "F:/MAFW/data/clips",
        "label_dir": "F:/MAFW/Train & Test Set/single/no_caption/set_1/train.txt",
    }
    preprocessed_data = run_preprocessing(config)
