"""Где два прохода трекера расходятся на одних и тех же рамках: кандидаты на перескок, которые стоит посмотреть глазами.

Для каждого номера готового прохода ищется момент, когда соответствующий номер другого прохода сменился, причём новый номер уже жил раньше на ком-то другом.
Сами по себе такие места не ошибка: в половине случаев номера перепутал другой проход. Лист со снимками вокруг каждого места: --sheet.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from mct.render import BACKGROUND, Stamps, paste
from mct.video import clock_text, read_selected_frames

BOX = ["x1", "y1", "x2", "y2"]
MATCH_IOU = 0.6
MIN_SIDE = 5  # наблюдений с каждой стороны, около 0.5 с
LIVED_FRAMES = 12  # новый номер жил раньше хотя бы столько кадров
OFFSETS_S = (-2.0, -1.0, -0.4, 0.1, 0.5, 1.0, 2.0)
SHEET_HEIGHT = 170


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1, y1 = np.maximum(a[:, None, 0], b[None, :, 0]), np.maximum(a[:, None, 1], b[None, :, 1])
    x2, y2 = np.minimum(a[:, None, 2], b[None, :, 2]), np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a, area_b = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-6)


def disagreements(base: pd.DataFrame, other: pd.DataFrame, camera: str) -> list[dict]:
    frames = {frame: part for frame, part in other.groupby("frame")}
    matched = []
    for frame, part in base.groupby("frame"):
        candidates = frames.get(frame)
        if candidates is None:
            continue
        iou = box_iou(part[BOX].to_numpy(float), candidates[BOX].to_numpy(float))
        best = iou.argmax(axis=1)
        good = iou[np.arange(len(part)), best] >= MATCH_IOU
        matched.append(pd.DataFrame({"frame": part.frame.to_numpy()[good], "raw_track": part.raw_track.to_numpy()[good], "t_s": part.t_s.to_numpy()[good],
                                     "other": candidates.raw_track.to_numpy()[best[good]]}))
    first_seen = other.groupby("raw_track").frame.min().to_dict()
    found = []
    for raw, part in pd.concat(matched, ignore_index=True).sort_values("frame").groupby("raw_track"):
        numbers = part.other.to_numpy()
        bounds = [0, *(np.flatnonzero(numbers[1:] != numbers[:-1]) + 1).tolist(), len(numbers)]
        segments = [(numbers[a], a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b - a >= MIN_SIDE]
        for (old, a0, a1), (new, b0, b1) in zip(segments[:-1], segments[1:]):
            lived = int(part.frame.iloc[b0]) - first_seen[new]
            if old != new and lived > LIVED_FRAMES:
                found.append({"camera": camera, "raw_track": int(raw), "t_s": round(float(part.t_s.iloc[b0]), 2), "time": clock_text(part.t_s.iloc[b0])[3:10],
                              "before": a1 - a0, "after": b1 - b0, "other_before": int(old), "other_after": int(new), "other_lived_frames": lived})
    return found


def draw_sheet(table: pd.DataFrame, bases: dict[str, pd.DataFrame], videos: Path, out: Path) -> None:
    picks, wanted = {}, {}
    for number, row in table.iterrows():
        part = bases[row.camera]
        part = part[part.raw_track == row.raw_track]
        chosen = [part.iloc[(part.t_s - (row.t_s + offset)).abs().argmin()] for offset in OFFSETS_S]
        picks[number] = [c for c, offset in zip(chosen, OFFSETS_S) if abs(c.t_s - (row.t_s + offset)) < 0.6]
        wanted.setdefault(row.camera, set()).update(int(c.frame) for c in picks[number])
    images = {(camera, frame): image for camera, frames in wanted.items() for frame, image in read_selected_frames(videos / f"{camera}.mkv", frames, (2592, 1944))}
    stamps = Stamps()
    width = 750
    sheet = np.full(((SHEET_HEIGHT + 34) * ((len(table) + 1) // 2), width * 2, 3), BACKGROUND, np.uint8)
    for number, row in table.iterrows():
        left, top = (number % 2) * width, (number // 2) * (SHEET_HEIGHT + 34)
        paste(sheet, stamps.text(f"{number + 1}. к{row.camera[-1]} номер {row.raw_track} {row.time}  другой проход {row.other_before}->{row.other_after}", 11, bold=True), left + 2, top + 1)
        x = left + 2
        for c in picks[number]:
            crop = images[(row.camera, int(c.frame))][int(c.y1 * 2):int(c.y2 * 2), int(c.x1 * 2):int(c.x2 * 2)]
            if not crop.size:
                continue
            crop = cv2.resize(crop, (max(int(crop.shape[1] * SHEET_HEIGHT / crop.shape[0]), 1), SHEET_HEIGHT), interpolation=cv2.INTER_AREA)[:, :110]
            if x + crop.shape[1] > left + width - 2:
                break
            sheet[top + 18:top + 18 + SHEET_HEIGHT, x:x + crop.shape[1]] = crop
            paste(sheet, stamps.text(f"{c.t_s - row.t_s:+.1f}", 9), x, top + 18 + SHEET_HEIGHT)
            x += crop.shape[1] + 3
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path, help="папка другого прохода, как artifacts/perception")
    parser.add_argument("--base", type=Path, default=Path("artifacts/perception"))
    parser.add_argument("--videos", type=Path, default=Path("data"))
    parser.add_argument("--sheet", type=Path, default=None)
    args = parser.parse_args()

    found, bases = [], {}
    for path in sorted(args.base.glob("camera_new_?.parquet")):
        bases[path.stem] = pd.read_parquet(path, columns=["frame", "raw_track", "t_s", *BOX])
        found += disagreements(bases[path.stem], pd.read_parquet(args.run / path.name, columns=["frame", "raw_track", *BOX]), path.stem)
    table = pd.DataFrame(found).sort_values("t_s").reset_index(drop=True)
    print(table.to_string())
    if args.sheet:
        draw_sheet(table, bases, args.videos, args.sheet)


if __name__ == "__main__":
    main()
