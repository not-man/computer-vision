"""Векторы внешности для уже найденных людей: другая модель, фон закрашен по маске, снимки чаще, исходное разрешение. Детектор не нужен.

Снимки те же, что в <камера>_emb.npz, или с --every каждое N-е наблюдение номера трекера. Рамки у края кадра не берутся, как в mct.perception.
С --full-res вырезки берутся из исходного кадра, вдвое крупнее рабочего: дальние люди чётче, а близкие сжимаются без наложения частот.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

from mct.embedder import crop_embedder, without_background
from mct.perception import PROCESS_SIZE
from mct.video import read_selected_frames

BATCH = 256
FULL_RES_SCALE = 2  # исходные 2592×1944 вдвое крупнее PROCESS_SIZE
MAX_CROP_HEIGHT = 400  # больше не нужно ни одной модели, а память так ограничена


def snapshots(perception: Path, camera: str, every: int | None) -> pd.DataFrame:
    table = pd.read_parquet(perception / f"{camera}.parquet", columns=["frame", "raw_track", "x1", "y1", "x2", "y2", "touches_edge", "mask_rle"])
    if every is None:
        keys = pd.DataFrame(np.load(perception / f"{camera}_emb.npz")["keys"], columns=["frame", "raw_track"])
        return keys.merge(table, on=["frame", "raw_track"], how="left")
    table = table[~table.touches_edge].sort_values(["raw_track", "frame"])
    return table[table.groupby("raw_track").cumcount() % every == 0].sort_values("frame").reset_index(drop=True)


def limited(crop: np.ndarray) -> np.ndarray:
    if crop.shape[0] <= MAX_CROP_HEIGHT:
        return crop.copy()
    size = (max(round(crop.shape[1] * MAX_CROP_HEIGHT / crop.shape[0]), 1), MAX_CROP_HEIGHT)
    return cv2.resize(crop, size, interpolation=cv2.INTER_AREA)


def read_crops(video: Path, rows: pd.DataFrame, full_res: bool = False) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Вырезки и маски в порядке строк rows. Из исходного кадра только вырезки: маски записаны в размере рабочего."""
    width, height = PROCESS_SIZE
    scale = FULL_RES_SCALE if full_res else 1
    crops: list = [None] * len(rows)
    masks: list = [None] * len(rows)
    by_frame = {frame: part for frame, part in rows.groupby("frame")}
    for frame, image in read_selected_frames(video, by_frame, (width * scale, height * scale)):
        for row in by_frame[frame].itertuples():
            x1, y1 = int(round(row.x1 * scale)), int(round(row.y1 * scale))
            x2, y2 = max(int(round(row.x2 * scale)), x1 + 1), max(int(round(row.y2 * scale)), y1 + 1)
            if full_res:
                crops[row.Index] = limited(image[y1:y2, x1:x2])
                continue
            crops[row.Index] = image[y1:y2, x1:x2].copy()
            masks[row.Index] = mask_utils.decode({"size": [height, width], "counts": row.mask_rle.encode("ascii")})[y1:y2, x1:x2]
    missing = sum(crop is None for crop in crops)
    if missing:
        raise RuntimeError(f"{video.name}: не прочитано снимков: {missing}")
    return crops, masks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", type=Path)
    parser.add_argument("--perception", type=Path, default=Path("artifacts/perception"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["osnet_ain_x1_0_msmt17"])
    parser.add_argument("--kinds", nargs="+", default=["mask"], choices=["box", "mask"])
    parser.add_argument("--every", type=int, default=None, help="каждое N-е наблюдение вместо записанных снимков")
    parser.add_argument("--suffix", default=None, help="писать <out>/<камера>_emb_<suffix>.npz вместо отдельной папки на вариант")
    parser.add_argument("--full-res", action="store_true", help="вырезки из исходного кадра, только --kinds box")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.full_res and "mask" in args.kinds:
        parser.error("--full-res работает только с --kinds box")

    cameras = sorted(path.stem for path in args.videos.glob("*.mkv"))
    data = {}
    for camera in cameras:
        rows = snapshots(args.perception, camera, args.every)
        data[camera] = (rows[["frame", "raw_track"]].to_numpy(np.int64), *read_crops(args.videos / f"{camera}.mkv", rows, args.full_res))
        print(camera, "снимков", len(rows), flush=True)
    for model in args.models:
        try:
            embedder = crop_embedder(model, device=args.device)
        except Exception as error:
            print(f"{model}: не загрузилась ({type(error).__name__}: {error})", flush=True)
            continue
        for kind in args.kinds:
            folder = args.out if args.suffix else args.out / f"{model.replace('/', '_')}_{kind}"
            folder.mkdir(parents=True, exist_ok=True)
            for camera, (keys, crops, masks) in data.items():
                target = folder / (f"{camera}_emb_{args.suffix}.npz" if args.suffix else f"{camera}_emb.npz")
                if target.exists():
                    continue
                inputs = crops if kind == "box" else [without_background(crop, mask) for crop, mask in zip(crops, masks)]
                vectors = np.vstack([embedder.embed_crops(inputs[start:start + BATCH]) for start in range(0, len(inputs), BATCH)])
                np.savez_compressed(target, keys=keys, vectors=vectors)
            print(model, kind, "готово", flush=True)


if __name__ == "__main__":
    main()
