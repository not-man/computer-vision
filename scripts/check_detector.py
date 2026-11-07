"""Проверка первого часа: скорость детектора и совпадение масок с ID трекера на реальном отрезке."""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from mct.video import read_frames

SOURCE_SIZE = (1296, 972)  # половина исходных 2592×1944


def mask_box_iou(mask: np.ndarray, box: np.ndarray) -> float:
    """Совпадение рамки трека с рамкой маски: при перепутанных ID оно падает к нулю."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0.0
    mx1, my1, mx2, my2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    x1, y1, x2, y2 = box
    inter = max(0.0, min(mx2, x2) - max(mx1, x1)) * max(0.0, min(my2, y2) - max(my1, y1))
    union = (mx2 - mx1) * (my2 - my1) + (x2 - x1) * (y2 - y1) - inter
    return float(inter / union) if union else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--model", default="yolo11s-seg.pt")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--out", type=Path, default=Path("artifacts/detector"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    source_fps = yaml.safe_load(Path("configs/clocks.yaml").read_text(encoding="utf-8"))[args.video.stem]["fps"]
    model = YOLO(args.model)
    ious, frames, with_people, overlays, ids = [], 0, 0, 0, set()
    infer_s = 0.0
    for _, frame in read_frames(args.video, SOURCE_SIZE, source_fps, args.fps, args.start, args.duration):
        started = time.perf_counter()
        result = model.track(
            frame, imgsz=args.imgsz, conf=args.conf, classes=[0], persist=True,
            tracker="botsort.yaml", retina_masks=True, device="cpu", verbose=False,
        )[0]
        infer_s += time.perf_counter() - started
        frames += 1
        if result.boxes is not None and result.boxes.id is not None and result.masks is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            masks = result.masks.data.cpu().numpy()
            assert len(boxes) == len(masks) == len(result.boxes.id)
            ious += [mask_box_iou(m, b) for m, b in zip(masks, boxes)]
            ids |= set(result.boxes.id.int().tolist())
        if len(result.boxes):
            with_people += 1
            if with_people % 40 == 1 and overlays < 4:
                cv2.imwrite(str(args.out / f"{args.video.stem}_{Path(args.model).stem}_{overlays}.jpg"), result.plot())
                overlays += 1

    ious = np.array(ious) if ious else np.zeros(1)
    print(json.dumps({
        "camera": args.video.stem,
        "model": args.model,
        "imgsz": args.imgsz,
        "frames": frames,
        "infer_fps": round(frames / infer_s, 2) if infer_s else None,
        "frames_with_people": with_people,
        "track_ids": len(ids),
        "mask_box_iou_median": round(float(np.median(ious)), 3),
        "mask_box_iou_p05": round(float(np.percentile(ious, 5)), 3),
    }))


if __name__ == "__main__":
    main()
