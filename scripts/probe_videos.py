"""Параметры записей и контрольные кадры: начало, середина и конец каждой камеры."""
import argparse
import json
import re
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg


def ffmpeg_info(path: Path) -> dict:
    # ffmpeg без выходного файла завершается с ошибкой, но печатает заголовок контейнера
    err = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(path)],
        capture_output=True, text=True,
    ).stderr
    stream = re.search(r"Stream #.*Video: (.*)", err)
    return {
        "duration": (m.group(1) if (m := re.search(r"Duration: ([\d:.]+|N/A)", err)) else None),
        "start": (m.group(1) if (m := re.search(r"start: ([\d.\-]+)", err)) else None),
        "stream": stream.group(1) if stream else None,
        "metadata": re.findall(r"^\s+(\w+)\s*: (.+)$", err.split("Stream #")[0], re.M),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", type=Path)
    parser.add_argument("--out", type=Path, default=Path("artifacts/probe"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    report = {}
    for path in sorted(args.videos.glob("*.mkv")):
        cap = cv2.VideoCapture(str(path))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        info = {
            "size_mb": round(path.stat().st_size / 1e6, 1),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frames": frames,
            **ffmpeg_info(path),
        }
        for name, pos in (("start", 0.02), ("mid", 0.5), ("end", 0.95)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frames * pos))
            ok, frame = cap.read()
            if ok:
                cv2.imwrite(str(args.out / f"{path.stem}_{name}.jpg"), frame)
        cap.release()
        report[path.stem] = info
    (args.out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
