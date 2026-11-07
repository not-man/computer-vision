"""Полный проход по камерам. Готовые камеры пропускаются, поэтому после обрыва просто запустить снова."""
import argparse
import json
from pathlib import Path

import yaml

from mct.embedder import create_embedder
from mct.perception import CONFIGS, process_camera


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", type=Path)
    parser.add_argument("--out", type=Path, default=Path("artifacts/perception"))
    parser.add_argument("--model", default="weights/yolo11s-seg.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.1)  # слабые детекции нужны трекеру, фильтр идёт позже
    parser.add_argument("--reid-model", default="osnet_ain_x1_0_msmt17")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--clock-start", type=float, default=None, help="начало окна по общим часам, секунды от 17:00:00; у каждой камеры свой первый кадр")
    parser.add_argument("--detect-scale", type=int, default=1, help="во сколько раз крупнее рабочего кадра кадр для детектора")
    parser.add_argument("--every-frame", action="store_true", help="трекер видит все кадры записи, в таблицы идут те же кадры, что при --fps")
    args = parser.parse_args()

    clocks = yaml.safe_load((CONFIGS / "clocks.yaml").read_text(encoding="utf-8"))

    embedder = create_embedder(args.device, args.reid_model)
    print(f"ReID: {embedder.name}", flush=True)
    for video in sorted(args.videos.glob("*.mkv")):
        if (args.out / f"{video.stem}_summary.json").exists():
            print(f"{video.stem}: уже готово", flush=True)
            continue
        start = args.start if args.clock_start is None else args.clock_start - clocks[video.stem]["clock_at_frame_0"]
        summary = process_camera(
            video, args.out, embedder, args.model, args.device,
            args.fps, args.imgsz, args.conf, start_s=start, duration_s=args.duration, detect_scale=args.detect_scale, every_frame=args.every_frame,
        )
        print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
