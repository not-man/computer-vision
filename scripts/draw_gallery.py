"""Галерея для ручной проверки: у каждого трека по снимку на треклет, отдельно всё, что отброшено.

В строке одного трека должен быть один человек, в галерее отброшенного не должно быть людей.
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from mct.geometry import IMAGE_SIZE
from mct.render import Stamps, paste
from mct.video import clock_text, read_selected_frames

THUMB_HEIGHT = 112
ROW_HEIGHT = 136
LEFT_WIDTH = 250
PAGE_WIDTH = 1500
ROWS_PER_PAGE = 22
HINT_CROPS = 4
GRID_CELL = (120, 150)
GRID_COLUMNS = 12
GRID_ROWS = 7
PAD = 6
BACKGROUND = (28, 28, 28)
TEXT = (235, 235, 235)
MUTED = (165, 165, 165)


def best_rows(table: pd.DataFrame, key: list[str]) -> pd.DataFrame:
    """Самый крупный вид: сначала не обрезанные краем кадра, затем по площади маски."""
    ordered = table.sort_values(["touches_edge", "mask_area"], ascending=[True, False])
    return ordered.groupby(key, sort=False).head(1)


def grab_crops(videos: Path, wanted: pd.DataFrame) -> dict[tuple[str, int, int], np.ndarray]:
    def one_camera(item: tuple[str, pd.DataFrame]) -> dict:
        camera, rows = item
        by_frame: dict[int, list] = {}
        for row in rows.itertuples():
            by_frame.setdefault(int(row.frame), []).append(row)
        crops = {}
        for number, frame in read_selected_frames(videos / f"{camera}.mkv", by_frame, IMAGE_SIZE):
            for row in by_frame[number]:
                x1, y1 = max(int(row.x1), 0), max(int(row.y1), 0)
                x2, y2 = min(int(np.ceil(row.x2)), IMAGE_SIZE[0]), min(int(np.ceil(row.y2)), IMAGE_SIZE[1])
                crops[(camera, number, int(row.raw_track))] = frame[y1:max(y2, y1 + 1), x1:max(x2, x1 + 1)].copy()
        return crops

    merged: dict = {}
    with ThreadPoolExecutor(5) as pool:
        for crops in pool.map(one_camera, wanted.groupby("camera")):
            merged.update(crops)
    return merged


def thumbnail(crop: np.ndarray, height: int, max_width: int) -> np.ndarray:
    scale = min(height / max(crop.shape[0], 1), max_width / max(crop.shape[1], 1))
    return cv2.resize(crop, (max(int(crop.shape[1] * scale), 1), max(int(crop.shape[0] * scale), 1)), interpolation=cv2.INTER_AREA)


def save_crops(best: pd.DataFrame, crops: dict, tracks_dir: Path) -> None:
    """Снимки для досье в интерфейсе: по одному на треклет."""
    out = tracks_dir / "crops"
    out.mkdir(exist_ok=True)
    index = {}
    for row in best.itertuples():
        crop = crops.get((row.camera, int(row.frame), int(row.raw_track)))
        if crop is not None:
            name = f"tracklet_{int(row.tracklet)}.jpg"
            cv2.imwrite(str(out / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            index[int(row.tracklet)] = {"file": name, "camera": row.camera, "t_s": round(float(row.t_s), 2)}
    (out / "index.json").write_text(json.dumps(index), encoding="utf-8")


def track_pages(result: dict, best: pd.DataFrame, crops: dict, out: Path, stamps: Stamps) -> int:
    rows_by_tracklet = {int(row.tracklet): row for row in best.itertuples()}
    pages = 0
    tracks = result["tracks"]
    for start in range(0, len(tracks), ROWS_PER_PAGE):
        chunk = tracks[start:start + ROWS_PER_PAGE]
        page = np.full((ROW_HEIGHT * len(chunk), PAGE_WIDTH, 3), 28, np.uint8)
        for index, track in enumerate(chunk):
            top = index * ROW_HEIGHT
            duration = (track["t_end_ms"] - track["t_start_ms"]) / 1000
            paste(page, stamps.text(f"#{track['track_id']}  {clock_text(track['t_start_ms'] / 1000)[:8]}", 15, TEXT, bold=True), PAD, top + 4)
            cameras = ",".join(camera[-1] for camera in track["cameras"])
            paste(page, stamps.text(f"{duration:.0f} с  cam {cameras}  {track['close_zone']}", 12, MUTED), PAD, top + 26)
            tiers = " ".join(f"{link['tier'][0].upper()}{link['g_m']:.1f}" + ("?" if link["confidence"] == "yellow" else "") for link in track["links"])
            paste(page, stamps.text(tiers, 12, MUTED, max_width=LEFT_WIDTH - 2 * PAD), PAD, top + 44)
            for hint in track["possibly_same_person_as"]:
                paste(page, stamps.text(f"похож на #{hint['track_id']} ({hint['r']:.2f})", 12, (220, 150, 220)), PAD, top + ROW_HEIGHT - 26)
            x = LEFT_WIDTH
            for tracklet in track["tracklets"]:
                row = rows_by_tracklet.get(tracklet)
                crop = None if row is None else crops.get((row.camera, int(row.frame), int(row.raw_track)))
                if crop is None:
                    continue
                thumb = thumbnail(crop, THUMB_HEIGHT, 120)
                if x + thumb.shape[1] > PAGE_WIDTH:
                    break
                paste(page, thumb, x, top + 2)
                paste(page, stamps.text(f"{row.camera[-1]} {clock_text(row.t_s)[3:8]}", 11, MUTED), x, top + THUMB_HEIGHT + 4)
                x += max(thumb.shape[1], 48) + PAD
            cv2.line(page, (0, top + ROW_HEIGHT - 1), (PAGE_WIDTH, top + ROW_HEIGHT - 1), (70, 70, 70), 1)
        pages += 1
        cv2.imwrite(str(out / f"tracks_{pages:02d}.jpg"), page, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return pages


def hints_page(result: dict, best: pd.DataFrame, crops: dict, out: Path, stamps: Stamps) -> int:
    """Пары Призрака рядом: слева ушедший трек, справа новый. Здесь сразу видно, тот ли это человек."""
    rows_by_tracklet = {int(row.tracklet): row for row in best.itertuples()}
    by_id = {track["track_id"]: track for track in result["tracks"]}
    pairs = [(track, hint) for track in result["tracks"] for hint in track["possibly_same_person_as"]]
    if not pairs:
        return 0

    def clearest(track: dict) -> list:
        rows = [rows_by_tracklet[i] for i in track["tracklets"] if i in rows_by_tracklet]
        return sorted(rows, key=lambda row: -row.mask_area)[:HINT_CROPS]

    page = np.full((ROW_HEIGHT * len(pairs), PAGE_WIDTH, 3), 28, np.uint8)
    for index, (track, hint) in enumerate(pairs):
        top = index * ROW_HEIGHT
        absent = int(round(hint["absent_s"]))
        paste(page, stamps.text(f"#{track['track_id']} ← #{hint['track_id']}", 15, TEXT, bold=True), PAD, top + 4)
        paste(page, stamps.text(f"R {hint['r']:.2f}, отсутствовал {absent // 60}:{absent % 60:02d}", 12, MUTED), PAD, top + 26)
        x = LEFT_WIDTH
        for side, shown in enumerate((by_id[hint["track_id"]], track)):
            for row in clearest(shown):
                crop = crops.get((row.camera, int(row.frame), int(row.raw_track)))
                if crop is None:
                    continue
                thumb = thumbnail(crop, THUMB_HEIGHT, 120)
                paste(page, thumb, x, top + 2)
                paste(page, stamps.text(f"{row.camera[-1]} {clock_text(row.t_s)[3:8]}", 11, MUTED), x, top + THUMB_HEIGHT + 4)
                x += max(thumb.shape[1], 48) + PAD
            if side == 0:
                cv2.rectangle(page, (x + 8, top + 2), (x + 12, top + ROW_HEIGHT - 6), (220, 150, 220), -1)
                x += 26
        cv2.line(page, (0, top + ROW_HEIGHT - 1), (PAGE_WIDTH, top + ROW_HEIGHT - 1), (70, 70, 70), 1)
    cv2.imwrite(str(out / "hints.jpg"), page, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return len(pairs)


def rejected_pages(rows: list[tuple[pd.Series, str]], crops: dict, out: Path, stamps: Stamps) -> int:
    pages = 0
    per_page = GRID_COLUMNS * GRID_ROWS
    for start in range(0, len(rows), per_page):
        page = np.full((GRID_ROWS * GRID_CELL[1], GRID_COLUMNS * GRID_CELL[0], 3), 28, np.uint8)
        for index, (row, reason) in enumerate(rows[start:start + per_page]):
            left, top = (index % GRID_COLUMNS) * GRID_CELL[0], (index // GRID_COLUMNS) * GRID_CELL[1]
            crop = crops.get((row.camera, int(row.frame), int(row.raw_track)))
            if crop is not None:
                paste(page, thumbnail(crop, GRID_CELL[1] - 34, GRID_CELL[0] - 4), left + 2, top + 2)
            paste(page, stamps.text(f"{row.camera[-1]} {clock_text(row.t_s)[3:8]} {row.conf:.2f}", 10, TEXT), left + 2, top + GRID_CELL[1] - 31)
            paste(page, stamps.text(reason, 10, MUTED), left + 2, top + GRID_CELL[1] - 16)
        pages += 1
        cv2.imwrite(str(out / f"rejected_{pages:02d}.jpg"), page, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return pages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("perception", type=Path)
    parser.add_argument("tracks", type=Path)
    parser.add_argument("--videos", type=Path, default=Path("data"))
    args = parser.parse_args()
    out = args.tracks / "gallery"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.jpg"):
        old.unlink()

    result = json.loads((args.tracks / "tracks.json").read_text(encoding="utf-8"))
    columns = ["frame", "raw_track", "x1", "y1", "x2", "y2", "mask_area", "conf", "touches_edge"]
    detections = pd.concat(
        [pd.read_parquet(path, columns=columns).assign(camera=path.stem) for path in sorted(args.perception.glob("camera_new_?.parquet"))],
        ignore_index=True,
    )
    located = pd.read_parquet(args.tracks / "observations.parquet", columns=["camera", "frame", "raw_track", "tracklet", "t_s"])
    assigned = located.merge(detections, on=["camera", "frame", "raw_track"])
    best = best_rows(assigned, ["tracklet"])

    # Отброшенное: фильтр ложных детекций и короткие треклеты без итогового трека
    kept = {tracklet for track in result["tracks"] for tracklet in track["tracklets"]}
    rejected: list[tuple[pd.Series, str]] = []
    short = best[~best.tracklet.isin(kept)]
    rejected += [(row, "короткий") for _, row in short.iterrows()]
    filtered = pd.read_csv(args.tracks / "dropped.csv")
    for row in filtered.itertuples():
        rows = detections[(detections.camera == row.camera) & (detections.raw_track == row.raw_track)
                          & (detections.frame >= row.frame_start) & (detections.frame <= row.frame_end)]
        if len(rows):
            chosen = best_rows(rows.assign(group=0), ["group"]).iloc[0].copy()
            chosen["t_s"] = row.t_start
            rejected.append((chosen, row.reason.split(":")[0]))
    rejected.sort(key=lambda item: (item[1], item[0].camera, item[0].t_s))

    wanted = pd.concat([best[["camera", "frame", "raw_track", "x1", "y1", "x2", "y2"]],
                        pd.DataFrame([item[0][["camera", "frame", "raw_track", "x1", "y1", "x2", "y2"]] for item in rejected])],
                       ignore_index=True).drop_duplicates(["camera", "frame", "raw_track"])
    crops = grab_crops(args.videos, wanted)

    save_crops(best, crops, args.tracks)
    stamps = Stamps()
    print("страниц треков:", track_pages(result, best, crops, out, stamps), " страниц отброшенного:", rejected_pages(rejected, crops, out, stamps),
          " подсказок:", hints_page(result, best, crops, out, stamps))


if __name__ == "__main__":
    main()
