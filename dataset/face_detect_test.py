import cv2
import numpy as np
from retinaface import RetinaFace

if __name__ == '__main__':
    video_dir = "F:/MAFW/data/clips/00019.mp4"
    cap = cv2.VideoCapture(video_dir)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    _, frame = cap.read()
    faces = RetinaFace.detect_faces(frame)

    if isinstance(faces, dict):
        face_keypoints = faces[list(faces.keys())[0]]['landmarks']
        left_eye = face_keypoints['left_eye']
        right_eye = face_keypoints['right_eye']
        nose = face_keypoints['nose']
        left_mouth = face_keypoints['mouth_left']
        right_mouth = face_keypoints['mouth_right']
        src_pts = np.float32([left_eye, right_eye, nose, left_mouth, right_mouth])

        desired_width, desired_height = 224, 224
        target_left_eye = [0.7 * desired_width, 0.35 * desired_height]
        target_right_eye = [0.3 * desired_width, 0.35 * desired_height]
        target_nose = [0.5 * desired_width, 0.6 * desired_height]
        target_left_mouth = [0.65 * desired_width, 0.8 * desired_height]
        target_right_mouth = [0.35 * desired_width, 0.8 * desired_height]

        dst_pts = np.float32([target_left_eye, target_right_eye, target_nose, target_left_mouth, target_right_mouth])
        M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        aligned_face = cv2.warpAffine(frame, M, (desired_width, desired_height))
        cv2.imshow('Aligned and Cropped Face', aligned_face)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
