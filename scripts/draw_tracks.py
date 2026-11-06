"""Ручная проверка треков: все камеры в один момент, маска цветом итогового трека, путь на плане."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from pycocotools import mask as mask_utils

from mct.geometry import IMAGE_SIZE
from mct.plan import Plan
from mct.render import assign_colors
from mct.video import read_frames

CELL = (648, 486)
PROCESS_FPS = 10.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("perception", type=Path)
    parser.add_argument("tracks", type=Path)
    parser.add_argument("--videos", type=Path, default=Path("../data"))
    parser.add_argument("--plan", type=Path, default=Path("../data/office_camera.png"))
    parser.add_argument("--clock-start", type=float, required=True)
    parser.add_argument("--clock-end", type=float, required=True)
    parser.add_argument("--step", type=float, default=3.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or args.tracks / "check"
    out.mkdir(parents=True, exist_ok=True)

    result = json.loads((args.tracks / "tracks.json").read_text())
    track_of = {tracklet: track["track_id"] for track in result["tracks"] for tracklet in track["tracklets"]}
    colors = assign_colors(result["tracks"])
    assigned = pd.read_parquet(args.tracks / "observations.parquet")
    assigned["track"] = assigned.tracklet.map(track_of)
    clocks = yaml.safe_load(Path("configs/clocks.yaml").read_text())
    times = np.arange(args.clock_start, args.clock_end + 1e-9, args.step)
    plan_image = cv2.imread(str(args.plan))
    scale = Plan.load().px_per_m

    panels: dict[str, dict[int, np.ndarray]] = {}
    for name, clock in clocks.items():
        video = args.videos / f"{name}.mkv"
        wanted = {int(round((t - clock["clock_at_frame_0"]) * clock["fps"])): i for i, t in enumerate(times)}
        if not video.exists() or not (args.perception / f"{name}.parquet").exists():
            continue
        masks = pd.read_parquet(args.perception / f"{name}.parquet")
        table = assigned[assigned.camera == name]
        panels[name] = {}
        step = clock["fps"] / PROCESS_FPS
        for n, frame in read_frames(video, IMAGE_SIZE, clock["fps"], PROCESS_FPS, threads=4):
            hits = [i for target, i in wanted.items() if abs(target - n) < step / 2 + 1e-9 and i not in panels[name]]
            if hits:
                image = frame.copy()
                for row in masks[masks.frame == n].itertuples():
                    match = table[(table.frame == n) & (table.raw_track == row.raw_track)]
                    track = match.track.iloc[0] if len(match) else np.nan
                    mask = mask_utils.decode({"size": [IMAGE_SIZE[1], IMAGE_SIZE[0]], "counts": row.mask_rle.encode()}) > 0
                    ys, xs = np.nonzero(mask)
                    if not len(xs):
                        continue
                    if np.isnan(track):
                        image[mask] = (0.6 * image[mask] + 0.4 * np.array((130, 130, 130))).astype(np.uint8)
                        label, color = f"x {row.conf:.2f}", (200, 200, 200)
                    else:
                        color = colors[int(track)]
                        image[mask] = (0.5 * image[mask] + 0.5 * np.array(color)).astype(np.uint8)
                        label = f"#{int(track)}"
                    anchor = (int(xs.min()), max(int(ys.min()) - 4, 26))
                    cv2.putText(image, label, anchor, 0, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
                    cv2.putText(image, label, anchor, 0, 0.9, color, 2, cv2.LINE_AA)
                cv2.putText(image, name[-5:], (8, IMAGE_SIZE[1] - 12), 0, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
                for i in hits:
                    panels[name][i] = cv2.resize(image, CELL)
            if n > max(wanted) + step:
                break

    for i, t in enumerate(times):
        plan = plan_image.copy()
        for track in result["tracks"]:
            path = [p for p in track["plan_path"] if p[2] / 1000 <= t]
            if len(path) < 1:
                continue
            points = (np.array(path)[:, :2] * scale).round().astype(np.int32)
            cv2.polylines(plan, [points], False, colors[track["track_id"]], 2, cv2.LINE_AA)
            if track["t_end_ms"] / 1000 >= t:
                cv2.circle(plan, tuple(int(v) for v in points[-1]), 6, colors[track["track_id"]], -1, cv2.LINE_AA)
                cv2.putText(plan, f"#{track['track_id']}", (int(points[-1][0]) + 6, int(points[-1][1])), 0, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        crop = plan[200:1160, 280:740]
        plan_cell = np.full((CELL[1], CELL[0], 3), 255, np.uint8)
        factor = min(CELL[0] / crop.shape[1], CELL[1] / crop.shape[0])
        small = cv2.resize(crop, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        plan_cell[:small.shape[0], :small.shape[1]] = small
        total = int(17 * 3600 + t)
        cv2.putText(plan_cell, f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}", (CELL[0] - 190, 40), 0, 1.0, (0, 0, 200), 2, cv2.LINE_AA)
        blank = np.zeros((CELL[1], CELL[0], 3), np.uint8)
        cells = [panels.get(name, {}).get(i, blank) for name in clocks] + [plan_cell]
        sheet = np.vstack([np.hstack(cells[:3]), np.hstack(cells[3:])])
        cv2.imwrite(str(out / f"t_{int(t)}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("кадров проверки:", len(times), out)


if __name__ == "__main__":
    main()
