import os
import sys
# from moviepy.editor import VideoFileClip, AudioFileClip
from pydub import AudioSegment

def extract_audio_from_mp4(mp4_file_path, output_dir="output_audio", sample_rate=4800):
    if not os.path.isfile(mp4_file_path):
        print(f"Error: File '{mp4_file_path}' does not exist.")
        return
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    try:
        filename = os.path.splitext(os.path.basename(mp4_file_path))[0]
        wav_file_path = os.path.join(output_dir, f"{filename}.flac")
        # video_clip = AudioFileClip(mp4_file_path)

        audio = AudioSegment.from_file(mp4_file_path, format="mp4")
        audio.set_frame_rate(sample_rate)
        audio.export(wav_file_path, format="flac")

        # if video_clip.audio is None:
        #     print(f"No audio found in '{mp4_file_path}'")
        #     return f"No audio found in '{mp4_file_path}'"
        # video_clip.audio.write_audiofile(wav_file_path, fps=sample_rate)
        print(f"Audio extracted and saved to '{wav_file_path}'")
        return wav_file_path
    except Exception as e:
        print(f"Error processing '{mp4_file_path}': {e}")
        return f"Error processing '{mp4_file_path}': {e}"


if __name__ == "__main__":
    file_path = 'F:/MAFW/data/clips'
    mp4_files = []
    for root, _, files in os.walk(file_path):
        for file in files:
            if file.endswith(".mp4"):
                filename = os.path.splitext(os.path.basename(file))[0]
                wav_file_path = os.path.join(root, f"{filename}.flac")
                if not os.path.exists(wav_file_path):
                    mp4_files.append(os.path.join(root, file))
    for mp4_file in mp4_files:
        extract_audio_from_mp4(mp4_file, output_dir=file_path)
