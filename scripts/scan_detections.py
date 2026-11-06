"""Детекции без трекинга на всей записи, 1 кадр/с и низкий порог: где возникают ложные срабатывания."""
import argparse
from pathlib import Path

import cv2
import pandas as pd
import torch
import yaml
from ultralytics import YOLO

from mct.perception import PROCESS_SIZE
from mct.video import read_frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--model", default="weights/yolo11s-seg.pt")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--out", type=Path, default=Path("artifacts/fp_scan"))
    args = parser.parse_args()

    torch.set_num_threads(3)
    camera = args.video.stem
    crops_dir = args.out / camera
    crops_dir.mkdir(parents=True, exist_ok=True)
    clock = yaml.safe_load(Path("configs/clocks.yaml").read_text())[camera]
    model = YOLO(args.model)

    rows = []
    for index, frame in read_frames(args.video, PROCESS_SIZE, clock["fps"], 1.0, threads=2):
        result = model.predict(
            frame, imgsz=args.imgsz, conf=args.conf, classes=[0], device="cpu", verbose=False,
        )[0]
        for number, (box, score) in enumerate(zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy())):
            x1, y1, x2, y2 = box.round().astype(int)
            crop_name = f"{index:06d}_{number}.jpg"
            cv2.imwrite(str(crops_dir / crop_name), frame[max(y1, 0):y2, max(x1, 0):x2])
            rows.append({
                "camera": camera, "frame": index,
                "t_s": clock["clock_at_frame_0"] + index / clock["fps"], "conf": float(score),
                "x1": float(box[0]), "y1": float(box[1]), "x2": float(box[2]), "y2": float(box[3]),
                "crop": crop_name,
            })
    pd.DataFrame(rows).to_parquet(args.out / f"{camera}.parquet", index=False)
    print(camera, "детекций:", len(rows))


if __name__ == "__main__":
    main()
