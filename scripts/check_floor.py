"""Проверка привязки на людях: рост, совпадение положений между камерами и сдвиг времени между ними."""
import argparse
import json
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from pycocotools import mask as mask_utils

from mct.geometry import IMAGE_SIZE, load_cameras

COLORS = {"camera_new_1": (0, 0, 255), "camera_new_2": (0, 160, 0), "camera_new_3": (255, 0, 0),
          "camera_new_4": (0, 140, 255), "camera_new_5": (200, 0, 200)}


def locate(frame: pd.DataFrame, camera, name: str) -> pd.DataFrame:
    rows = []
    size = [IMAGE_SIZE[1], IMAGE_SIZE[0]]
    for row in frame.itertuples():
        if row.conf < 0.5 or row.touches_edge:
            continue
        mask = mask_utils.decode({"size": size, "counts": row.mask_rle.encode()})
        ys, xs = np.nonzero(mask)
        if len(xs) < 200:
            continue
        foot, head = camera.foot_and_head(np.column_stack([xs, ys]))
        floor = camera.rectified_to_floor(foot)[0]
        rows.append({
            "camera": name, "t_s": row.t_s, "track": row.raw_track, "X": floor[0], "Y": floor[1],
            "person_height": camera.height_of(foot, head),
        })
    return pd.DataFrame(rows)


def nearest_distances(a: pd.DataFrame, b: pd.DataFrame, lag_s: float = 0.0, window_s: float = 0.06) -> np.ndarray:
    """Для каждого наблюдения камеры a расстояние до ближайшего наблюдения камеры b в тот же момент."""
    b_times = b.t_s.to_numpy() + lag_s
    b_xy = b[["X", "Y"]].to_numpy()
    out = []
    for t, x, y in a[["t_s", "X", "Y"]].itertuples(index=False):
        near = np.abs(b_times - t) <= window_s
        if near.any():
            out.append(float(np.hypot(*(b_xy[near] - (x, y)).T).min()))
    return np.array(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("perception", type=Path)
    parser.add_argument("--plan", type=Path, default=Path("data/office_camera.png"))
    parser.add_argument("--features", type=Path, default=Path("configs/plan_features.yaml"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/geometry/people_on_plan.jpg"))
    parser.add_argument("--max-obs", type=int, default=20000)
    args = parser.parse_args()

    scale = yaml.safe_load(args.features.read_text(encoding="utf-8"))["plan_px_per_m"]
    cameras = load_cameras()
    located = {}
    for name, camera in cameras.items():
        path = args.perception / f"{name}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        if len(frame) > args.max_obs:
            frame = frame.sample(args.max_obs, random_state=0).sort_values("t_s")
        located[name] = locate(frame, camera, name)
        heights = located[name].person_height
        print(name, json.dumps({
            "наблюдений": len(heights),
            "рост_медиана_м": round(float(heights.median()), 2),
            "рост_p10_p90_м": [round(float(heights.quantile(q)), 2) for q in (0.1, 0.9)],
        }, ensure_ascii=False), flush=True)

    for first, second in combinations(located, 2):
        a, b = located[first], located[second]
        now = nearest_distances(a, b)
        if len(now) < 20:
            print(f"{first[-1]}–{second[-1]}: общих моментов мало ({len(now)})", flush=True)
            continue
        # Сдвиг времени оцениваем по доле совпадений: медиана ближайшего обманывает, когда камеры видят разных людей
        lags = np.arange(-8.0, 8.01, 0.2)
        shares = np.array([
            float((d < 0.7).mean()) if len(d := nearest_distances(a, b, lag)) >= 20 else 0.0 for lag in lags
        ])
        best = int(np.argmax(shares))
        shifted = np.concatenate([nearest_distances(a, b, lag) for lag in (-5.0, 5.0)])
        print(
            f"{first[-1]}–{second[-1]}: сравнений {len(now)}, медиана {np.median(now):.2f} м, совпало (< 0.7 м) {float((now < 0.7).mean()):.2f}, "
            f"при сдвиге ±5 с {float((shifted < 0.7).mean()):.2f}; лучший сдвиг {lags[best]:+.1f} с, совпало {shares[best]:.2f}",
            flush=True,
        )

    plan = cv2.imread(str(args.plan))
    for name, table in located.items():
        for x, y in (table[["X", "Y"]].to_numpy() * scale).round().astype(int):
            if 0 <= x < plan.shape[1] and 0 <= y < plan.shape[0]:
                cv2.circle(plan, (int(x), int(y)), 2, COLORS[name], -1, cv2.LINE_AA)
    for name, camera in cameras.items():
        center = (round(camera.x * scale), round(camera.y * scale))
        cv2.drawMarker(plan, center, COLORS[name], cv2.MARKER_TILTED_CROSS, 16, 2, cv2.LINE_AA)
        cv2.putText(plan, name[-1], (center[0] + 8, center[1] - 6), 0, 0.7, COLORS[name], 2, cv2.LINE_AA)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), plan, [cv2.IMWRITE_JPEG_QUALITY, 90])


if __name__ == "__main__":
    main()
