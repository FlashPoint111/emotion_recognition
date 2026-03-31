import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def extract_audio_fast(input_file, output_file, sample_rate=48000):
    try:
        command = [
            'ffmpeg',
            '-y',
            '-i', str(input_file),
            '-vn',
            '-ar', str(sample_rate),  # 这里严格保证了输出采样率为 48000
            '-c:a', 'flac',
            '-loglevel', 'error',
            str(output_file)
        ]

        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return True, str(input_file)

    except Exception as e:
        return False, f"{input_file} | Error: {str(e)}"


def main():
    base_dir = Path('F:/DFEW/Clip/original')
    target_sample_rate = 48000

    print("Scanning directory...")
    tasks = []

    for mp4_file in base_dir.rglob('*.mp4'):
        flac_file = base_dir / f"{mp4_file.stem}.flac"

        if not flac_file.exists():
            tasks.append((mp4_file, flac_file))

    total_tasks = len(tasks)
    if total_tasks == 0:
        print("No files to process.")
        return

    print(f"Found {total_tasks} files. Starting parallel processing...")

    optimal_workers = min(32, (os.cpu_count() or 4) + 4)
    success_count = 0

    with ThreadPoolExecutor(max_workers=optimal_workers) as executor:
        future_to_file = {
            executor.submit(extract_audio_fast, in_f, out_f, target_sample_rate): in_f
            for in_f, out_f in tasks
        }

        for future in as_completed(future_to_file):
            success, result_msg = future.result()
            if success:
                success_count += 1
                print(f"[{success_count}/{total_tasks}] Processed: {Path(result_msg).name}")
            else:
                print(f"[FAILED] {result_msg}")

    print(f"\nProcessing complete. Success: {success_count}/{total_tasks}")


if __name__ == "__main__":
    main()