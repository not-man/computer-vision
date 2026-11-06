"""Показания часов до и после каждого разрыва записи при последовательном чтении: идёт ли время в разрыв."""
import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np

from mct.video import FFMPEG

W, H = 2592, 1944
BOXES = {"camera_new_4": (1950, 20, 642, 90)}
DEFAULT_BOX = (0, 0, 760, 60)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--out", type=Path, default=Path("artifacts/clocks"))
    args = parser.parse_args()

    x, y, w, h = BOXES.get(args.video.stem, DEFAULT_BOX)
    pts = np.load(args.out / f"{args.video.stem}_pts.npy") / 1000.0
    gaps = np.nonzero(np.diff(pts) > 1.0)[0]
    # Пакеты идут в порядке записи: перед разрывом, после разрыва, а ещё начало и конец
    wanted = sorted({0, *[int(g) for g in gaps], *[int(g) + 1 for g in gaps], len(pts) - 1})
    cmd = [
        FFMPEG, "-v", "error", "-threads", "3", "-i", str(args.video), "-map", "0:v", "-fps_mode", "passthrough",
        "-vf", f"crop={w}:{h}:{x}:{y}", "-pix_fmt", "gray", "-f", "rawvideo", "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    rows, index = [], 0
    while len(buf := proc.stdout.read(w * h)) == w * h:
        if index in wanted:
            cell = cv2.resize(np.frombuffer(buf, np.uint8).reshape(h, w), (640, 60))
            label = np.full((60, 200), 0, np.uint8)
            cv2.putText(label, f"pts {pts[index]:.2f}", (4, 40), 0, 0.8, 255, 2)
            rows.append(np.hstack([label, cell]))
        index += 1
    proc.wait()
    print(args.video.stem, "кадров при последовательном чтении:", index, "пакетов:", len(pts))
    cv2.imwrite(str(args.out / f"{args.video.stem}_gaps.png"), np.vstack(rows))


if __name__ == "__main__":
    main()
