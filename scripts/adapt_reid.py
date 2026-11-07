"""Дообучает OSNet-AIN на самой записи и считает векторы. Подписи: scripts/make_adapt_labels.py. Запускается в Colab на GPU.

Варианты: вся запись и две половины. Модель половины проверяется на другой половине, которой не видела: это честная оценка.
Размеченная толпа не входит ни в один вариант. Пишет <out>/<вариант>_<шагов>/<камера>_emb.npz, как scripts/reembed.py, и веса варианта на всю запись.
Готовые варианты пропускаются, поэтому после обрыва просто запустить снова.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from mct.adapt import INPUT_SIZE, ChainSampler, embed_all, recalibrate_norms, train
from mct.embedder import BoxmotEmbedder
from reembed import read_crops, snapshots

SEED = 20261107
MIDDLE_S = 2921.6  # середина записи по общим часам
MARGIN_S = 30.0  # у половин пауза вокруг середины: один человек не попадёт по обе стороны в один проход
FOLDS = {"all": (-np.inf, np.inf), "first": (-np.inf, MIDDLE_S - MARGIN_S), "second": (MIDDLE_S + MARGIN_S, np.inf)}


def read_camera(videos: Path, perception: Path, camera: str, every: int, limit: int | None) -> tuple[pd.DataFrame, np.ndarray]:
    rows = snapshots(perception, camera, every)
    if limit is not None:
        rows = rows.iloc[:limit].reset_index(drop=True)
    crops, _ = read_crops(videos / f"{camera}.mkv", rows, full_res=True)
    resized = [cv2.resize(crop, INPUT_SIZE, interpolation=cv2.INTER_AREA if crop.shape[0] > INPUT_SIZE[1] else cv2.INTER_LINEAR)[:, :, ::-1] for crop in crops]
    return rows[["frame", "raw_track"]].assign(camera=camera), np.stack(resized)


def write_vectors(folder: Path, keys: pd.DataFrame, vectors: np.ndarray) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for camera, part in keys.groupby("camera"):
        np.savez_compressed(folder / f"{camera}_emb.npz", keys=part[["frame", "raw_track"]].to_numpy(np.int64), vectors=vectors[part.index.to_numpy()])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", type=Path)
    parser.add_argument("--perception", type=Path, default=Path("artifacts/perception"))
    parser.add_argument("--labels", type=Path, default=Path("artifacts/adapt"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--every", type=int, default=3)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[500, 1500, 3000])
    parser.add_argument("--device", default="0")
    parser.add_argument("--limit", type=int, default=None, help="только первые снимки каждой камеры: для короткой проверки")
    args = parser.parse_args()
    device = f"cuda:{args.device}" if args.device.isdigit() else args.device

    cameras = sorted(path.stem for path in args.videos.glob("*.mkv"))
    with ThreadPoolExecutor(len(cameras)) as pool:
        parts = list(pool.map(lambda camera: read_camera(args.videos, args.perception, camera, args.every, args.limit), cameras))
    keys = pd.concat([part[0] for part in parts], ignore_index=True)
    images = torch.from_numpy(np.concatenate([part[1] for part in parts])).permute(0, 3, 1, 2)
    print("снимков", len(keys), flush=True)

    labels = pd.read_parquet(args.labels / "labels.parquet").drop_duplicates(["camera", "frame", "raw_track"])
    table = keys.merge(labels, on=["camera", "frame", "raw_track"], how="left")
    numbers = {chain: number for number, chain in enumerate(sorted(table.chain.dropna().unique()))}
    chains = table.chain.map(numbers).fillna(-1).to_numpy(np.int64)
    different = np.zeros((len(numbers), len(numbers)), bool)
    for pair in pd.read_csv(args.labels / "different.csv").itertuples():
        if pair.a in numbers and pair.b in numbers:
            different[numbers[pair.a], numbers[pair.b]] = different[numbers[pair.b], numbers[pair.a]] = True
    print("снимков с подписью", int((chains >= 0).sum()), "цепочек", len(numbers), "чужих пар", int(different.sum()) // 2, flush=True)

    start = BoxmotEmbedder("osnet_ain_x1_0_msmt17", device=args.device)._encoder._runtime.model
    state = deepcopy(start.state_dict())
    torch.manual_seed(SEED)

    plain = args.out / "adabn"
    if not (plain / f"{cameras[-1]}_emb.npz").exists():
        recalibrate_norms(start, images, device, seed=SEED)
        write_vectors(plain, keys, embed_all(start, images, device))
        start.load_state_dict(state)
        print("статистика нормировок обновлена", flush=True)

    for fold, (low, high) in FOLDS.items():
        if all((args.out / f"{fold}_{steps}" / f"{cameras[-1]}_emb.npz").exists() for steps in args.checkpoints):
            print(fold, "уже готово", flush=True)
            continue
        inside = (table.t_s >= low) & (table.t_s < high)
        fold_chains = np.where(inside.to_numpy(), chains, -1)
        sampler = ChainSampler(fold_chains, table.camera.to_numpy(), different, SEED)
        print(fold, "снимков", int((fold_chains >= 0).sum()), "цепочек", len(sampler.all), flush=True)
        start.load_state_dict(state)

        def on_checkpoint(steps: int) -> None:
            write_vectors(args.out / f"{fold}_{steps}", keys, embed_all(start, images, device))
            if fold == "all":
                torch.save(start.state_dict(), args.out / f"osnet_ain_adapted_{steps}.pt")
            print(fold, steps, "векторы записаны", flush=True)

        train(start, images, sampler, torch.from_numpy(different), torch.from_numpy(fold_chains.clip(0)), max(args.checkpoints), device, on_checkpoint, tuple(args.checkpoints))
    print("ГОТОВО", flush=True)


if __name__ == "__main__":
    main()
