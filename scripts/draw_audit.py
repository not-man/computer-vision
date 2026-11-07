"""Листы для проверки глазами по критериям организаторов. Выборка случайная, с постоянным зерном, порядок её не зависит от проверяющего.

Переходы: пары снимков кусков, между которыми алгоритм провёл связь. Закрытия: окрестность места, где трек закончился, чуть до и после конца.
Рядом с листами пишутся таблицы с пустой колонкой вердикта. Листы содержат людей: как и записи, их нельзя выкладывать открыто.
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from mct.render import BACKGROUND, Stamps, paste
from mct.video import clock_text, read_selected_frames

TRANSITIONS = {"sustained": 30, "brief": 15, "gap": 10, "return": 99}  # возвраты решает одна внешность, поэтому смотрим все
CLOSURES = {"DOOR": 10, "OPEN": 99, "EDGE_OUT": 99, "STATIONARY": 99, "OCCLUDER": 99, "RECORDING_END": 99}  # редкие виды целиком, двери выборкой
SEED = 20261107
CROP_HEIGHT = 150
PAIRS_PER_ROW = 4
ROWS_PER_SHEET = 5
FRAME_SIZE = (1296, 972)
CELL = (214, 160)  # окрестность места конца трека на листе закрытий
AROUND_PX = 150
MOMENTS_S = (-0.5, 1.0, 3.0)
TRACKS_PER_SHEET = 12


def sample(items: list, count: int, rng: random.Random, everything: bool = False) -> list:
    if everything or len(items) <= count:
        return sorted(items, key=lambda item: item["t_s"])
    return sorted(rng.sample(items, count), key=lambda item: item["t_s"])


def transitions_table(tracks: list[dict], everything: bool = False) -> pd.DataFrame:
    rng = random.Random(SEED)
    by_tier: dict[str, list[dict]] = {tier: [] for tier in TRANSITIONS}
    for track in tracks:
        for link in track["links"]:
            if link["tier"] in by_tier and (link["camera_a"] != link["camera_b"] or link["tier"] in ("gap", "return")):
                by_tier[link["tier"]].append({
                    "track": track["track_id"], "tier": link["tier"], "a": link["a"], "b": link["b"], "camera_a": link["camera_a"][-1], "camera_b": link["camera_b"][-1],
                    "t_s": link["t_ms"] / 1000, "time": clock_text(link["t_ms"] / 1000)[:8], "g_m": link["g_m"], "contact_s": link["contact_s"], "r": link["r"], "confidence": link["confidence"],
                })
    rows = [row for tier, count in TRANSITIONS.items() for row in sample(by_tier[tier], count, rng, everything)]
    print({tier: len(items) for tier, items in by_tier.items()})
    return pd.DataFrame(rows).assign(verdict="", note="")


def thumbnail(path: Path, height: int) -> np.ndarray:
    image = cv2.imread(str(path))
    scale = height / image.shape[0]
    resized = cv2.resize(image, (max(int(image.shape[1] * scale), 1), height), interpolation=cv2.INTER_AREA)
    return resized[:, :int(height * 0.85)]


def draw_transitions(table: pd.DataFrame, crops_dir: Path, out: Path) -> None:
    index = json.loads((crops_dir / "index.json").read_text(encoding="utf-8"))
    stamps = Stamps()
    width = 1440 // PAIRS_PER_ROW
    for start in range(0, len(table), PAIRS_PER_ROW * ROWS_PER_SHEET):
        part = table.iloc[start:start + PAIRS_PER_ROW * ROWS_PER_SHEET]
        sheet = np.full(((CROP_HEIGHT + 40) * ROWS_PER_SHEET, width * PAIRS_PER_ROW, 3), BACKGROUND, np.uint8)
        for n, row in enumerate(part.itertuples()):
            left, top = (n % PAIRS_PER_ROW) * width, (n // PAIRS_PER_ROW) * (CROP_HEIGHT + 40)
            paste(sheet, stamps.text(f"{row.Index + 1}. #{row.track} {row.tier} к{row.camera_a}-к{row.camera_b} {row.time[3:]} R{row.r if row.r is not None else 0:.2f}", 11, bold=True), left + 2, top + 2)
            x = left + 2
            for tracklet in (row.a, row.b):
                item = index.get(str(tracklet))
                if item is None:
                    continue
                crop = thumbnail(crops_dir / item["file"], CROP_HEIGHT)
                sheet[top + 20:top + 20 + crop.shape[0], x:x + crop.shape[1]] = crop
                x += crop.shape[1] + 6
        name = out / f"transitions_{start // (PAIRS_PER_ROW * ROWS_PER_SHEET) + 1}.jpg"
        cv2.imwrite(str(name), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(name.name, sheet.shape)


def closures_table(tracks: list[dict], observations: pd.DataFrame, owner: dict[int, int], everything: bool = False) -> pd.DataFrame:
    rng = random.Random(SEED)
    by_zone: dict[str, list[dict]] = {}
    last_rows = observations[observations.source != "unreliable"].assign(track=lambda t: t.tracklet.map(owner)).dropna(subset=["track"]).sort_values("t_s").groupby("track").tail(1)
    last = {int(row.track): row for row in last_rows.itertuples()}
    for track in tracks:
        row = last[track["track_id"]]
        by_zone.setdefault(track["close_zone"], []).append({
            "track": track["track_id"], "close": track["close_zone"], "camera": row.camera, "t_s": row.t_s, "time": clock_text(row.t_s)[:8],
            "x1": row.x1, "y1": row.y1, "x2": row.x2, "y2": row.y2,
        })
    print({zone: len(items) for zone, items in by_zone.items()})
    rows = [row for zone, items in sorted(by_zone.items()) if zone in CLOSURES for row in sample(items, CLOSURES[zone], rng, everything)]
    return pd.DataFrame(rows).assign(verdict="", note="")


def draw_closures(table: pd.DataFrame, videos: Path, clocks: dict, out: Path) -> None:
    wanted: dict[str, dict[int, list[tuple[int, int]]]] = {}
    frame_of = lambda camera, t: int(round((t - clocks[camera]["clock_at_frame_0"]) * clocks[camera]["fps"]))
    for row in table.itertuples():
        for moment, delta in enumerate(MOMENTS_S):
            frame = min(frame_of(row.camera, row.t_s + delta), clocks[row.camera]["frames"] - 1)
            wanted.setdefault(row.camera, {}).setdefault(frame, []).append((row.Index, moment))
    cells: dict[tuple[int, int], np.ndarray] = {}
    for camera, frames in wanted.items():
        for number, image in read_selected_frames(videos / f"{camera}.mkv", frames, FRAME_SIZE):
            for row_index, moment in frames[number]:
                row = table.loc[row_index]
                center_x, center_y = int((row.x1 + row.x2) / 2), int((row.y1 + row.y2) / 2)
                half_w, half_h = AROUND_PX, int(AROUND_PX * CELL[1] / CELL[0])
                x1, x2 = np.clip([center_x - half_w, center_x + half_w], 0, FRAME_SIZE[0])
                y1, y2 = np.clip([center_y - half_h, center_y + half_h], 0, FRAME_SIZE[1])
                view = image.copy()
                if moment == 0:
                    cv2.rectangle(view, (int(row.x1), int(row.y1)), (int(row.x2), int(row.y2)), (0, 255, 255), 1)
                cells[(row_index, moment)] = cv2.resize(view[y1:y2, x1:x2], CELL, interpolation=cv2.INTER_AREA)
    stamps = Stamps()
    for start in range(0, len(table), TRACKS_PER_SHEET):
        part = table.iloc[start:start + TRACKS_PER_SHEET]
        rows = (TRACKS_PER_SHEET + 1) // 2
        sheet = np.full(((CELL[1] + 22) * rows, (CELL[0] + 2) * len(MOMENTS_S) * 2 + 10, 3), BACKGROUND, np.uint8)
        for n, row in enumerate(part.itertuples()):
            left, top = (n % 2) * ((CELL[0] + 2) * len(MOMENTS_S) + 10), (n // 2) * (CELL[1] + 22)
            paste(sheet, stamps.text(f"{row.Index + 1}. #{row.track} {row.close} к{row.camera[-1]} {row.time[3:]}", 11, bold=True), left + 2, top + 2)
            for moment in range(len(MOMENTS_S)):
                cell = cells.get((row.Index, moment))
                if cell is not None:
                    sheet[top + 20:top + 20 + CELL[1], left + moment * (CELL[0] + 2):left + moment * (CELL[0] + 2) + CELL[0]] = cell
        name = out / f"closures_{start // TRACKS_PER_SHEET + 1}.jpg"
        cv2.imwrite(str(name), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(name.name, sheet.shape)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=["transitions", "closures"])
    parser.add_argument("--tracks", type=Path, default=Path("artifacts/tracks"))
    parser.add_argument("--videos", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/test-results/audit"))
    parser.add_argument("--all", action="store_true", help="не выборку, а все связи или закрытия")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    tracks = json.loads((args.tracks / "tracks.json").read_text(encoding="utf-8"))["tracks"]
    if args.what == "transitions":
        table = transitions_table(tracks, args.all)
        table.to_csv(args.out / "transitions_blank.csv", index=False)
        draw_transitions(table, args.tracks / "crops", args.out)
    else:
        owner = {tracklet: track["track_id"] for track in tracks for tracklet in track["tracklets"]}
        table = closures_table(tracks, pd.read_parquet(args.tracks / "observations.parquet"), owner, args.all)
        table.to_csv(args.out / "closures_blank.csv", index=False)
        draw_closures(table, args.videos, yaml.safe_load(Path("configs/clocks.yaml").read_text(encoding="utf-8")), args.out)


if __name__ == "__main__":
    main()
