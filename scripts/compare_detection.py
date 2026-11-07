"""Сравнение прогонов детектора на окнах записи: artifacts/trial/<вариант>/<окно>/ против готовой детекции на том же окне.

Склейка считается заново на одном окне и для варианта, и для готовой детекции, поэтому числа сравнимы только между собой.
Разметка толпы переносится на новый прогон по совпадению рамок. Векторы у обоих раз в секунду.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from mct import trinity as tr
from evaluation.fragment_truth import label_rows, score_fixed_truth, track_on_truth
from mct.geometry import load_cameras
from mct.plan import Plan
from mct.tracklets import build_tracklets, filter_config, load_perception
from scripts.port_labels import transfer_truth_rows

WINDOWS = {"crowd": (2700.0, 70.0), "door": (3095.0, 35.0)}
FAR_M = 20.0
MATCH_IOU = 0.5
NEW_IOU = 0.2


def box_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1, y1 = np.maximum(a[:, None, 0], b[None, :, 0]), np.maximum(a[:, None, 1], b[None, :, 1])
    x2, y2 = np.minimum(a[:, None, 2], b[None, :, 2]), np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-6)


def carry_over(old: pd.DataFrame, new: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Для строк new: человек из разметки old и наибольшее совпадение рамки с любой строкой old в том же кадре."""
    people = pd.Series(None, index=new.index, dtype=object)
    best = pd.Series(0.0, index=new.index)
    old = old.assign(person=label_rows(old))
    groups = {key: part for key, part in old.groupby(["camera", "frame"])}
    box = ["x1", "y1", "x2", "y2"]
    for key, part in new.groupby(["camera", "frame"]):
        other = groups.get(key)
        if other is None:
            continue
        iou = box_iou_matrix(part[box].to_numpy(float), other[box].to_numpy(float))
        nearest = iou.argmax(axis=1)
        values = iou[np.arange(len(part)), nearest]
        best.loc[part.index] = values
        people.loc[part.index] = np.where(values >= MATCH_IOU, other.person.to_numpy(dtype=object)[nearest], None)
    return people, best


def window_rows(observations: pd.DataFrame, embeddings: dict, start: float, duration: float) -> tuple[pd.DataFrame, dict]:
    chosen = observations[(observations.t_s >= start) & (observations.t_s < start + duration)]
    limited = {}
    for camera, value in embeddings.items():
        frames = chosen[chosen.camera == camera].frame.unique()
        inside = np.isin(value[0][:, 0], frames)
        limited[camera] = value[0][inside], value[1][inside]
    return chosen, limited


def run_window(observations: pd.DataFrame, embeddings: dict, zones: tr.ZoneMap) -> dict:
    tracklets, assigned, dropped = build_tracklets(observations, embeddings, filter_config())
    result = tr.link_tracklets(tracklets, tr.TrinityConfig(min_track_s=1.0), zones)
    owner = {i: t.track_id for t in result.tracks for i in t.tracklets}
    return dict(tracks=len(result.tracks), pieces=len(tracklets), dropped=len(dropped), assigned=assigned, track=assigned.tracklet.map(owner))


def describe(name: str, observations: pd.DataFrame, run: dict, cameras: dict, labels: dict | None = None, extra: str = "") -> None:
    distance = np.hypot(observations.foot_x - observations.camera.map(lambda c: cameras[c].x), observations.foot_y - observations.camera.map(lambda c: cameras[c].y))
    good = observations.conf >= 0.25
    text = (f"{name:30s} строк {len(observations):6d}, уверенных {int(good.sum()):6d}, дальше {FAR_M:.0f} м {int((good & (distance >= FAR_M)).sum()):5d}, "
            f"номеров трекера {observations.groupby('camera').raw_track.nunique().sum():4d}, кусков {run['pieces']:3d}, треков {run['tracks']:3d}")
    if labels is not None:
        text += f", толпа {labels['соответствие']:.3f}/{labels['чистота']:.3f} ({labels['в_треках']}/{labels['размечено']})"
    print(text + extra, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", type=Path, default=Path("artifacts/trial"))
    parser.add_argument("--perception", type=Path, default=Path("artifacts/perception"))
    parser.add_argument("--new-rows", type=Path, default=None, help="куда записать таблицу рамок, которых не было в готовой детекции")
    args = parser.parse_args()

    cameras, plan = load_cameras(), Plan.load()
    extra = [tr.Zone(**zone) for zone in (yaml.safe_load(Path("configs/zones.yaml").read_text(encoding="utf-8")) or {}).get("zones") or []]
    zones = tr.ZoneMap(plan.door_zones() + extra, strict_blind=False)
    base_observations, base_embeddings = load_perception(args.perception, cameras, plan, use_dense=False)
    for window, (start, duration) in WINDOWS.items():
        print(f"=== окно {window}: {start:.0f}–{start + duration:.0f}")
        old_rows, old_embeddings = window_rows(base_observations, base_embeddings, start, duration)
        old_run = run_window(old_rows, old_embeddings, zones)
        baseline_truth = transfer_truth_rows({camera: (rows, rows) for camera, rows in old_rows.groupby("camera")}) if window == "crowd" else None
        labels = (score_fixed_truth(baseline_truth.person, track_on_truth(baseline_truth, old_run["assigned"], old_run["track"], target="raw_track"))
                  if baseline_truth is not None else None)
        describe("готовая детекция", old_rows, old_run, cameras, labels)
        for folder in sorted(path for path in args.trial.glob("*") if (path / window).is_dir()):
            if not list((folder / window).glob("*_summary.json")):
                continue
            new_observations, new_embeddings = load_perception(folder / window, cameras, plan, use_dense=False)
            new_rows, new_limited = window_rows(new_observations, new_embeddings, start, duration)
            new_run = run_window(new_rows, new_limited, zones)
            _, best = carry_over(old_rows, new_run["assigned"])
            if window == "crowd":
                cameras_in_window = sorted(old_rows.camera.unique())
                transferred = transfer_truth_rows({camera: (old_rows[old_rows.camera == camera], new_rows[new_rows.camera == camera])
                                                   for camera in cameras_in_window})
                labels = score_fixed_truth(transferred.person,
                                           track_on_truth(transferred, new_run["assigned"], new_run["track"], target="new_raw_track"))
            else:
                labels = None
            kept = new_run["assigned"][new_run["track"].notna()]
            fresh = kept[best.loc[kept.index] < NEW_IOU]
            old_kept = old_run["assigned"][old_run["track"].notna()]
            lost = old_kept[carry_over(new_rows.assign(person=None), old_kept)[1] < NEW_IOU]
            describe(folder.name, new_rows, new_run, cameras, labels, f", новых рамок в треках {len(fresh)}, потеряно {len(lost)}")
            if args.new_rows is not None and len(fresh):
                args.new_rows.mkdir(parents=True, exist_ok=True)
                fresh.to_parquet(args.new_rows / f"{folder.name}_{window}.parquet", index=False)
                lost.to_parquet(args.new_rows / f"{folder.name}_{window}_lost.parquet", index=False)


if __name__ == "__main__":
    main()
