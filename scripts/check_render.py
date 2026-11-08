"""Проверка рендера: минутные части мозаики из всех камер по общим часам, размер и скорость записи."""
import argparse
import json
import time
from itertools import islice
from pathlib import Path

import cv2
import numpy as np
import yaml

from mct.video import H264Writer, read_frames

CELL = (426, 320)  # 4:3, как у камер; 3×2 ячейки при ширине 1280
CANVAS = (1280, 720)
FPS = 10.0


def clock_label(clock_s: float) -> str:
    total = int(clock_s) + 17 * 3600
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}.{int(clock_s * 10) % 10}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", type=Path)
    parser.add_argument("--clock-start", type=float, default=2490.0)
    parser.add_argument("--chunk-s", type=int, default=60)
    parser.add_argument("--chunks", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("render/check"))
    args = parser.parse_args()

    clocks = yaml.safe_load(Path("configs/clocks.yaml").read_text(encoding="utf-8"))
    readers = []
    for video in sorted(args.videos.glob("*.mkv")):
        clock = clocks[video.stem]
        readers.append((video.stem, clock, read_frames(
            video, CELL, clock["fps"], FPS, args.clock_start - clock["clock_at_frame_0"], threads=1,
        )))

    report = []
    for chunk in range(args.chunks):
        path = args.out / f"{chunk:03d}.mp4"
        started = time.perf_counter()
        frames = 0
        with H264Writer(path, CANVAS, FPS) as writer:
            for step in range(int(args.chunk_s * FPS)):
                canvas = np.zeros((CANVAS[1], CANVAS[0], 3), np.uint8)
                spread = []
                for position, (name, clock, reader) in enumerate(readers):
                    index, frame = next(reader)
                    clock_s = clock["clock_at_frame_0"] + index / clock["fps"]
                    spread.append(clock_s)
                    row, column = position // 3, position % 3
                    y, x = row * CELL[1], column * (CELL[0] + 1)
                    canvas[y:y + CELL[1], x:x + CELL[0]] = frame
                    cv2.putText(canvas, name[-5:], (x + 6, y + CELL[1] - 8), 0, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                label = f"{clock_label(min(spread))}   chunk {chunk}   spread {max(spread) - min(spread):.2f}s"
                cv2.putText(canvas, label, (10, CANVAS[1] - 30), 0, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                writer.write(canvas)
                frames += 1
        seconds = time.perf_counter() - started
        report.append({
            "file": str(path), "mb": round(path.stat().st_size / 1e6, 2),
            "frames": frames, "render_fps": round(frames / seconds, 1),
        })
        print(json.dumps(report[-1]), flush=True)
    for _, _, reader in readers:
        list(islice(reader, 0))
        reader.close()


if __name__ == "__main__":
    main()
