import cv2
import numpy as np
import pandas as pd

from mct.highlight import HIGHLIGHT, highlight_clip, track_boxes, visible_window
from mct.render import CANVAS, CELL, CELL_ORIGINS, FPS
from mct.video import H264Writer

CAMERAS = [f"camera_new_{index}" for index in range(1, 6)]


def observations() -> pd.DataFrame:
    return pd.DataFrame({
        "camera": ["camera_new_2", "camera_new_2", "camera_new_3"], "t_s": [100.0, 100.1, 100.1],
        "x1": [648.0, 648.0, 0.0], "y1": [486.0, 486.0, 0.0], "x2": [972.0, 972.0, 100.0], "y2": [729.0, 729.0, 100.0],
        "tracklet": [7, 7, 8],
    })


def test_boxes_land_in_the_cell_of_their_camera() -> None:
    boxes = track_boxes(observations(), [7], CAMERAS)
    assert len(boxes) == 2
    left, top = CELL_ORIGINS[1]
    assert boxes.x1.iloc[0] == left + CELL[0] / 2 and boxes.y1.iloc[0] == top + CELL[1] / 2


def test_window_covers_only_the_visible_part_of_the_minute() -> None:
    boxes = track_boxes(observations(), [7], CAMERAS)
    assert visible_window(boxes, 90.0, 150.0) == (99.0, 101.1)
    assert visible_window(boxes, 150.0, 210.0) is None


def test_clip_draws_the_track_and_keeps_the_rest(tmp_path) -> None:
    chunk = tmp_path / "chunk.mp4"
    with H264Writer(chunk, CANVAS, FPS) as writer:
        for _ in range(5):
            writer.write(np.full((CANVAS[1], CANVAS[0], 3), 40, np.uint8))
    boxes = track_boxes(observations(), [7], CAMERAS)
    out = highlight_clip(chunk, 100.0, boxes, 100.0, 100.3, tmp_path / "track.mp4", "#7")
    capture = cv2.VideoCapture(str(out))
    ok, frame = capture.read()
    capture.release()
    assert ok
    left, top = (int(round(value - 3)) for value in (boxes.x1.iloc[0], boxes.y1.iloc[0]))
    edge = frame[top + 1, left + 20].astype(int)
    assert np.abs(edge - np.array(HIGHLIGHT)).max() < 60
    assert np.abs(frame[5, 5].astype(int) - 40).max() < 15
