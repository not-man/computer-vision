"""Доля времени с движением по каждой камере: сколько кадров реально нужно детектору."""
import argparse
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np

W, H, FPS = 320, 240, 5
PIXEL_DELTA = 25  # яркость, ниже которой разница считается шумом
TOP_MASK = 16  # строки с наложенным таймкодом, он меняется каждую секунду


def scan(path: Path) -> np.ndarray:
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-threads", "3", "-i", str(path),
        "-vf", f"fps={FPS},scale={W}:{H}", "-pix_fmt", "gray", "-f", "rawvideo", "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    prev, fractions = None, []
    while len(buf := proc.stdout.read(W * H)) == W * H:
        frame = np.frombuffer(buf, np.uint8).reshape(H, W)[TOP_MASK:].astype(np.int16)
        if prev is not None:
            fractions.append(float((np.abs(frame - prev) > PIXEL_DELTA).mean()))
        prev = frame
    proc.wait()
    return np.array(fractions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--out", type=Path, default=Path("artifacts/activity"))
    parser.add_argument("--min-fraction", type=float, default=0.001)
    parser.add_argument("--pad-s", type=float, default=2.0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    fractions = scan(args.video)
    active = fractions > args.min_fraction
    pad = int(args.pad_s * FPS)
    padded = np.convolve(active, np.ones(2 * pad + 1), "same") > 0
    np.save(args.out / f"{args.video.stem}.npy", fractions)
    print(json.dumps({
        "camera": args.video.stem,
        "seconds": round(len(fractions) / FPS, 1),
        "active_share": round(float(active.mean()), 4),
        "active_share_padded": round(float(padded.mean()), 4),
        "active_seconds_padded": round(float(padded.sum()) / FPS, 1),
    }))


if __name__ == "__main__":
    main()
