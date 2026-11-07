"""Векторы внешности для уже найденных людей: другая модель, фон закрашен по маске, снимки чаще. Детектор не нужен.

Снимки те же, что в <камера>_emb.npz, или с --every каждое N-е наблюдение номера трекера. Рамки у края кадра не берутся, как в mct.perception.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

from mct.embedder import BoxmotEmbedder, without_background
from mct.perception import PROCESS_SIZE
from mct.video import read_selected_frames

BATCH = 256


def snapshots(perception: Path, camera: str, every: int | None) -> pd.DataFrame:
    table = pd.read_parquet(perception / f"{camera}.parquet", columns=["frame", "raw_track", "x1", "y1", "x2", "y2", "touches_edge", "mask_rle"])
    if every is None:
        keys = pd.DataFrame(np.load(perception / f"{camera}_emb.npz")["keys"], columns=["frame", "raw_track"])
        return keys.merge(table, on=["frame", "raw_track"], how="left")
    table = table[~table.touches_edge].sort_values(["raw_track", "frame"])
    return table[table.groupby("raw_track").cumcount() % every == 0].sort_values("frame").reset_index(drop=True)


def read_crops(video: Path, rows: pd.DataFrame) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Вырезки и маски в порядке строк rows."""
    width, height = PROCESS_SIZE
    crops: list = [None] * len(rows)
    masks: list = [None] * len(rows)
    by_frame = {frame: part for frame, part in rows.groupby("frame")}
    for frame, image in read_selected_frames(video, by_frame, PROCESS_SIZE):
        for row in by_frame[frame].itertuples():
            x1, y1 = int(round(row.x1)), int(round(row.y1))
            x2, y2 = max(int(round(row.x2)), x1 + 1), max(int(round(row.y2)), y1 + 1)
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
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cameras = sorted(path.stem for path in args.videos.glob("*.mkv"))
    data = {}
    for camera in cameras:
        rows = snapshots(args.perception, camera, args.every)
        data[camera] = (rows[["frame", "raw_track"]].to_numpy(np.int64), *read_crops(args.videos / f"{camera}.mkv", rows))
        print(camera, "снимков", len(rows), flush=True)
    for model in args.models:
        try:
            embedder = BoxmotEmbedder(model, device=args.device)
        except Exception as error:
            print(f"{model}: не загрузилась ({type(error).__name__}: {error})", flush=True)
            continue
        for kind in args.kinds:
            folder = args.out / f"{model}_{kind}"
            folder.mkdir(parents=True, exist_ok=True)
            for camera, (keys, crops, masks) in data.items():
                if (folder / f"{camera}_emb.npz").exists():
                    continue
                inputs = crops if kind == "box" else [without_background(crop, mask) for crop, mask in zip(crops, masks)]
                vectors = np.vstack([embedder.embed_crops(inputs[start:start + BATCH]) for start in range(0, len(inputs), BATCH)])
                np.savez_compressed(folder / f"{camera}_emb.npz", keys=keys, vectors=vectors)
            print(model, kind, "готово", flush=True)


if __name__ == "__main__":
    main()
