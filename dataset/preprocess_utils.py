import random

import cv2
import librosa
import numpy as np
from retinaface import RetinaFace
from PIL import Image


def get_face(video_dir, frames, res):
    target_left_eye = [0.7 * res, 0.35 * res]
    target_right_eye = [0.3 * res, 0.35 * res]
    target_nose = [0.5 * res, 0.6 * res]
    target_left_mouth = [0.65 * res, 0.8 * res]
    target_right_mouth = [0.35 * res, 0.8 * res]
    dst_pts = np.float32([target_left_eye, target_right_eye, target_nose, target_left_mouth, target_right_mouth])
    result = []

    cap = cv2.VideoCapture(video_dir)
    for idx in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        _, frame = cap.read()
        faces = RetinaFace.detect_faces(frame)
        face_keypoints = faces[list(faces.keys())[0]]['landmarks']
        left_eye = face_keypoints['left_eye']
        right_eye = face_keypoints['right_eye']
        nose = face_keypoints['nose']
        left_mouth = face_keypoints['mouth_left']
        right_mouth = face_keypoints['mouth_right']
        src_pts = np.float32([left_eye, right_eye, nose, left_mouth, right_mouth])
        M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        result.append(Image.fromarray(cv2.warpAffine(frame, M, (res, res))))

    return result

def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return total_frames, fps


def get_audio_length(audio_path, sr):
    try:
        audio, sr = librosa.load(audio_path, sr=sr)
        return len(audio)
    except Exception as e:
        print(f"Error loading audio from {audio_path}: {e}")
        return None


def check_requirements(total_frames, required_frames, audio_length, required_audio_length):
    return total_frames >= required_frames and audio_length >= required_audio_length


def get_random_start_time(duration_in_seconds, required_audio_length):
    max_start_time = duration_in_seconds - required_audio_length
    if max_start_time > 0:
        return random.randint(0, max_start_time)
    else:
        return 0
