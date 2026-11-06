"""Чистый фон камеры: медиана кадров со всей записи убирает проходящих людей."""
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from mct.video import FFMPEG

PROCESS_SIZE = (1296, 972)  # те же координаты, что у наблюдений в mct.perception


def grab(path: Path, t: float) -> np.ndarray | None:
    # Точное время здесь неважно, поэтому перемотка по меткам файла допустима
    width, height = PROCESS_SIZE
    cmd = [
        FFMPEG, "-v", "fatal", "-threads", "2", "-ss", str(t), "-i", str(path), "-frames:v", "1",
        "-vf", f"scale={width}:{height}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]
    buf = subprocess.run(cmd, capture_output=True).stdout
    if len(buf) != width * height * 3:
        return None
    return np.frombuffer(buf, np.uint8).reshape(height, width, 3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", type=Path)
    parser.add_argument("--samples", type=int, default=41)
    parser.add_argument("--out", type=Path, default=Path("artifacts/geometry/background"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    times = np.linspace(10, 1180, args.samples)
    for video in sorted(args.videos.glob("*.mkv")):
        with ThreadPoolExecutor(6) as pool:
            frames = [f for f in pool.map(lambda t: grab(video, t), times) if f is not None]
        cv2.imwrite(str(args.out / f"{video.stem}.png"), np.median(np.stack(frames), axis=0).astype(np.uint8))
        print(video.stem, "кадров:", len(frames), flush=True)


if __name__ == "__main__":
    main()
