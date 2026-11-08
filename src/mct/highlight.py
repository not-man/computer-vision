"""Видео выбранного трека: та же мозаика, но рамка трека обведена на всех камерах.

Ролики не перерисовываются: кадры берутся из готовой минуты, поверх рисуются рамки из observations.parquet.
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from mct.geometry import IMAGE_SIZE
from mct.render import CANVAS, CELL, CELL_ORIGINS, FPS
from mct.video import H264Writer

HIGHLIGHT = (0, 230, 255)  # жёлтый в BGR, на силуэтах других треков его нет
NEAR_S = 0.1  # как в мозаике: камера показывает ближайший кадр не дальше этого
PAD_S = 1.0  # до появления и после ухода показываем ещё секунду


def track_boxes(observations: pd.DataFrame, tracklets: list[int], cameras: list[str]) -> pd.DataFrame:
    """Рамки трека в пикселях мозаики: время, левый верхний и правый нижний угол."""
    rows = observations[observations.tracklet.isin(tracklets) & observations.camera.isin(cameras)]
    origin = {camera: CELL_ORIGINS[index] for index, camera in enumerate(cameras)}
    sx, sy = CELL[0] / IMAGE_SIZE[0], CELL[1] / IMAGE_SIZE[1]
    left = rows.camera.map(lambda camera: origin[camera][0]).to_numpy()
    top = rows.camera.map(lambda camera: origin[camera][1]).to_numpy()
    return pd.DataFrame({
        "t": rows.t_s.to_numpy(), "camera": rows.camera.to_numpy(),
        "x1": left + rows.x1.to_numpy() * sx, "y1": top + rows.y1.to_numpy() * sy,
        "x2": left + rows.x2.to_numpy() * sx, "y2": top + rows.y2.to_numpy() * sy,
    }).sort_values("t").reset_index(drop=True)


def visible_window(boxes: pd.DataFrame, chunk_start_s: float, chunk_end_s: float) -> tuple[float, float] | None:
    """Отрезок минуты, где трек виден, с запасом в секунду. None, если в этой минуте его нет."""
    inside = boxes[(boxes.t >= chunk_start_s) & (boxes.t < chunk_end_s)]
    if inside.empty:
        return None
    return max(chunk_start_s, float(inside.t.min()) - PAD_S), min(chunk_end_s, float(inside.t.max()) + PAD_S)


def draw_boxes(frame: np.ndarray, boxes: pd.DataFrame, t: float, label: str) -> None:
    near = boxes[(boxes.t - t).abs() <= NEAR_S]
    for _, row in near.assign(away=(near.t - t).abs()).sort_values("away").drop_duplicates("camera").iterrows():
        x1, y1, x2, y2 = (int(round(value)) for value in (row.x1 - 3, row.y1 - 3, row.x2 + 3, row.y2 + 3))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 5, cv2.LINE_AA)
        cv2.rectangle(frame, (x1, y1), (x2, y2), HIGHLIGHT, 3, cv2.LINE_AA)
        cv2.putText(frame, label, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, label, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, HIGHLIGHT, 1, cv2.LINE_AA)


def highlight_clip(chunk: Path, chunk_start_s: float, boxes: pd.DataFrame, t_from: float, t_to: float, out: Path, label: str) -> Path:
    """Пишет в out кусок минуты от t_from до t_to с обведённым треком."""
    first, last = int(round((t_from - chunk_start_s) * FPS)), int(round((t_to - chunk_start_s) * FPS))
    capture = cv2.VideoCapture(str(chunk))
    try:
        with H264Writer(out, CANVAS, FPS) as writer:
            number = 0
            while number <= last:
                ok, frame = capture.read()
                if not ok:
                    break
                if number >= first:
                    draw_boxes(frame, boxes, chunk_start_s + number / FPS, label)
                    writer.write(frame)
                number += 1
    finally:
        capture.release()
    return out
