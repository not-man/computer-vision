"""Полный проход по камерам. Готовые камеры пропускаются, поэтому после обрыва просто запустить снова."""
import argparse
import json
from pathlib import Path

from mct.embedder import create_embedder
from mct.perception import process_camera


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
    args = parser.parse_args()

    embedder = create_embedder(args.device, args.reid_model)
    print(f"ReID: {embedder.name}", flush=True)
    for video in sorted(args.videos.glob("*.mkv")):
        if (args.out / f"{video.stem}_summary.json").exists():
            print(f"{video.stem}: уже готово", flush=True)
            continue
        summary = process_camera(
            video, args.out, embedder, args.model, args.device,
            args.fps, args.imgsz, args.conf, start_s=args.start, duration_s=args.duration,
        )
        print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
