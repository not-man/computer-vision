"""Сверить вручную отмеченные точки на исходных кадрах с детекцией и сквозными ID."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import linear_sum_assignment

from evaluation.fragment_truth import score_fixed_truth
from mct.video import sampled_frame_index

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "evaluation" / "holdout_points_v1.csv"
IMAGE_SIZE = (1296, 972)
ANNOTATION_SIZE = (900, 675)


def read_points(path: Path = LABELS) -> pd.DataFrame:
    points = pd.read_csv(path)
    clocks = yaml.safe_load((ROOT / "configs" / "clocks.yaml").read_text(encoding="utf-8"))
    points["raw_frame"] = [round((row.clock_s - clocks[row.camera]["clock_at_frame_0"]) * clocks[row.camera]["fps"])
                           for row in points.itertuples()]
    def sampled(camera: str, raw_frame: int) -> int:
        step = clocks[camera]["fps"] / 10.0
        near = round(raw_frame / step)
        return min((sampled_frame_index(k, step) for k in range(max(0, near - 1), near + 2)),
                   key=lambda frame: (abs(frame - raw_frame), frame))
    points["frame"] = [sampled(row.camera, row.raw_frame) for row in points.itertuples()]
    points["x"] = points.x_900 * IMAGE_SIZE[0] / ANNOTATION_SIZE[0]
    points["y"] = points.y_675 * IMAGE_SIZE[1] / ANNOTATION_SIZE[1]
    return points


def read_detections(folder: Path, cameras: list[str]) -> pd.DataFrame:
    columns = ["camera", "frame", "raw_track", "x1", "y1", "x2", "y2"]
    return pd.concat([pd.read_parquet(folder / f"{camera}.parquet", columns=columns) for camera in cameras], ignore_index=True)


def match_points(points: pd.DataFrame, detections: pd.DataFrame) -> pd.Series:
    """Одно наблюдение не может отвечать за двух размеченных людей в одном кадре."""
    matched = pd.Series(pd.NA, index=points.index, dtype="Int64")
    for (camera, frame), group in points.groupby(["camera", "frame"], sort=False):
        candidates = detections[(detections.camera == camera) & (detections.frame == frame)].reset_index(drop=True)
        if candidates.empty:
            continue
        boxes = candidates[["x1", "y1", "x2", "y2"]].to_numpy(float)
        xy = group[["x", "y"]].to_numpy(float)
        inside = ((xy[:, None, 0] >= boxes[None, :, 0]) & (xy[:, None, 0] <= boxes[None, :, 2]) &
                  (xy[:, None, 1] >= boxes[None, :, 1]) & (xy[:, None, 1] <= boxes[None, :, 3]))
        center = (boxes[:, :2] + boxes[:, 2:]) / 2
        size = np.maximum(boxes[:, 2:] - boxes[:, :2], 1.0)
        distance = np.linalg.norm((xy[:, None, :] - center[None, :, :]) / size[None, :, :], axis=2)
        costs = np.where(inside, distance, 2.0)
        costs = np.column_stack([costs, np.ones((len(group), len(group)))])
        rows, columns = linear_sum_assignment(costs)
        for row, column in zip(rows, columns):
            if column < len(candidates) and costs[row, column] < 1.0:
                matched.loc[group.index[row]] = int(candidates.iloc[column].raw_track)
    return matched


def evaluate(points: pd.DataFrame, detections: pd.DataFrame, assigned: pd.DataFrame, owner: dict[int, int]) -> tuple[dict, pd.DataFrame]:
    raw = match_points(points, detections)
    if assigned.duplicated(["camera", "frame", "raw_track"]).any():
        raise ValueError("Повтор номера трекера в одном кадре")
    source = assigned[["camera", "frame", "raw_track", "tracklet"]].copy()
    source["track"] = source.tracklet.map(owner)
    lookups = source.set_index(["camera", "frame", "raw_track"])["track"]
    predicted = []
    for row, raw_track in zip(points.itertuples(), raw):
        predicted.append(lookups.get((row.camera, row.frame, int(raw_track)), np.nan) if pd.notna(raw_track) else np.nan)
    details = points[["camera", "frame", "person"]].copy()
    details["raw_track"] = raw
    details["track"] = pd.Series(predicted, index=points.index)
    result = score_fixed_truth(points.person, details.track)
    result["найдены_детекции"] = int(raw.notna().sum())
    return result, details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--perception", type=Path, default=ROOT / "artifacts" / "perception")
    parser.add_argument("--tracks", type=Path, default=ROOT / "artifacts" / "tracks")
    args = parser.parse_args()
    points = read_points()
    detections = read_detections(args.perception, sorted(points.camera.unique()))
    assigned = pd.read_parquet(args.tracks / "observations.parquet", columns=["camera", "frame", "raw_track", "tracklet"])
    tracks = json.loads((args.tracks / "tracks.json").read_text(encoding="utf-8"))["tracks"]
    owner = {piece: track["track_id"] for track in tracks for piece in track["tracklets"]}
    score, details = evaluate(points, detections, assigned, owner)
    print(score)
    print(details.to_string(index=False))


if __name__ == "__main__":
    main()
