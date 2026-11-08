"""Раскадровка кусков по времени: строка на кусок, столбец на момент. Показывает, где номер меняет человека."""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from mct.geometry import load_cameras
from mct.plan import Plan
from mct.render import BACKGROUND, Stamps, paste
from mct.tracklets import build_tracklets, filter_config, load_perception
from mct.trinity import TrinityConfig, Zone, ZoneMap, link_tracklets
from mct.video import clock_text, read_selected_frames


def board(camera: str, assigned: pd.DataFrame, owner: dict, t0: float, t1: float, step: float, out: Path, cell=(58, 118), video_dir=Path("data")) -> None:
    rows = assigned[(assigned.camera == camera) & (assigned.t_s >= t0) & (assigned.t_s <= t1) & assigned.tracklet.isin(list(owner))].copy()
    times = np.arange(t0, t1 + 1e-9, step)
    wanted = {}
    table = []
    for tracklet, g in rows.groupby("tracklet"):
        cells = []
        for t in times:
            i = (g.t_s - t).abs().argmin()
            r = g.iloc[i]
            if abs(r.t_s - t) <= step / 2 + 0.05:
                wanted.setdefault(int(r.frame), []).append(r)
                cells.append(r)
            else:
                cells.append(None)
        table.append((g.t_s.min(), int(tracklet), owner[int(tracklet)], int(g.raw_track.iloc[0]), cells))
    table.sort()
    frames = dict(read_selected_frames(video_dir / f"{camera}.mkv", sorted(wanted), (2592, 1944)))
    stamps = Stamps()
    width, height = cell
    label = 62
    sheet = np.full(((height + 2) * len(table) + 18, label + (width + 2) * len(times), 3), BACKGROUND, np.uint8)
    for j, t in enumerate(times):
        if j % 2 == 0:
            paste(sheet, stamps.text(clock_text(t)[6:10], 10), label + j * (width + 2), 2)
    for i, (_, tracklet, track, raw, cells) in enumerate(table):
        top = 18 + i * (height + 2)
        paste(sheet, stamps.text(f"#{track}", 13, bold=True), 2, top + 2)
        paste(sheet, stamps.text(f"к{tracklet}", 10), 2, top + 22)
        for j, r in enumerate(cells):
            if r is None or int(r.frame) not in frames:
                continue
            x1, y1, x2, y2 = [int(v * 2) for v in (r.x1, r.y1, r.x2, r.y2)]
            crop = frames[int(r.frame)][max(y1, 0):max(y2, 1), max(x1, 0):max(x2, 1)]
            if crop.size == 0:
                continue
            s = min(width / crop.shape[1], height / crop.shape[0])
            crop = cv2.resize(crop, (max(int(crop.shape[1] * s), 1), max(int(crop.shape[0] * s), 1)), interpolation=cv2.INTER_AREA)
            left = label + j * (width + 2)
            sheet[top:top + crop.shape[0], left:left + crop.shape[1]] = crop
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(out.name, sheet.shape, "кусков", len(table))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("start", type=float, help="секунды от 17:00:00")
    parser.add_argument("end", type=float)
    parser.add_argument("--cameras", default="1,2,3,4,5")
    parser.add_argument("--out", type=Path, default=Path("artifacts/boards"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plan = Plan.load()
    observations, embeddings = load_perception(Path("artifacts/perception"), load_cameras(), plan)
    tracklets, assigned, _ = build_tracklets(observations, embeddings, filter_config())
    extra = [Zone(**zone) for zone in (yaml.safe_load(Path("configs/zones.yaml").read_text(encoding="utf-8")) or {}).get("zones") or []]
    result = link_tracklets(tracklets, TrinityConfig(), ZoneMap(plan.door_zones() + extra, strict_blind=False))
    owner = {i: track.track_id for track in result.tracks for i in track.tracklets}
    for camera in args.cameras.split(","):
        board(f"camera_new_{camera}", assigned, owner, args.start, args.end, 1.5, args.out / f"board_{camera}.jpg", video_dir=Path("data"))


if __name__ == "__main__":
    main()
