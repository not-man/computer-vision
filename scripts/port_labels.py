"""Переносит ручную разметку на новый прогон детектора: номера трекера меняются, а рамки тех же людей остаются на месте.

Строка разметки и точка проверки ищут в том же кадре рамку нового прогона с наибольшим совпадением. С --write заменяет таблицы в evaluation/fragment_truth.py.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.fragment_truth import DIFFERENT_PEOPLE, LABELS, SAME_PERSON, TIME_TOLERANCE_S

BOX = ["x1", "y1", "x2", "y2"]
MATCH_IOU = 0.5
NEAR_S = (2.0, 5.0, 10.0)  # точка проверки может попасть в кадр без рамки: ищем всё дальше по времени
TRUTH = Path(__file__).resolve().parents[1] / "evaluation" / "fragment_truth.py"


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1, y1 = np.maximum(a[:, None, 0], b[None, :, 0]), np.maximum(a[:, None, 1], b[None, :, 1])
    x2, y2 = np.minimum(a[:, None, 2], b[None, :, 2]), np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-6)


def matched_numbers(old: pd.DataFrame, new: pd.DataFrame) -> pd.Series:
    """Для каждой строки old: номер трекера нового прогона с тем же человеком или NaN."""
    numbers = pd.Series(np.nan, index=old.index)
    groups = {frame: part for frame, part in new.groupby("frame")}
    for frame, part in old.groupby("frame"):
        other = groups.get(frame)
        if other is None:
            continue
        iou = box_iou(part[BOX].to_numpy(float), other[BOX].to_numpy(float))
        nearest = iou.argmax(axis=1)
        good = iou[np.arange(len(part)), nearest] >= MATCH_IOU
        numbers.loc[part.index[good]] = other.raw_track.to_numpy()[nearest[good]]
    return numbers


def port_labels(tables: dict[str, tuple[pd.DataFrame, pd.DataFrame]]) -> tuple[list[tuple], int, int]:
    ported, total, kept = [], 0, 0
    for camera, raw_track, start, end, person in LABELS:
        old, new = tables[camera]
        rows = old[(old.raw_track == raw_track) & (old.t_s >= start - TIME_TOLERANCE_S) & (old.t_s <= end + TIME_TOLERANCE_S)].sort_values("t_s")
        numbers = matched_numbers(rows, new)
        total += len(rows)
        kept += int(numbers.notna().sum())
        for number, part in rows[numbers.notna()].groupby(numbers[numbers.notna()]):
            ported.append((camera, int(number), round(float(part.t_s.min()), 2), round(float(part.t_s.max()), 2), person))
    return sorted(ported), total, kept


def port_point(tables: dict[str, tuple[pd.DataFrame, pd.DataFrame]], point: tuple) -> tuple | None:
    camera, raw_track, t = point
    old, new = tables[camera]
    own = old[old.raw_track == raw_track]
    for near in NEAR_S:
        rows = own[(own.t_s - t).abs() <= near].copy()
        if rows.empty:
            continue
        rows = rows.assign(away=(rows.t_s - t).abs()).sort_values("away")
        numbers = matched_numbers(rows, new)
        if numbers.notna().any():
            first = numbers.dropna().index[0]
            return camera, int(numbers[first]), round(float(rows.loc[first, "t_s"]), 1)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    columns = ["frame", "t_s", "raw_track", *BOX]
    tables = {}
    for path in sorted(args.old.glob("camera_new_?.parquet")):
        target = args.new / path.name
        if target.exists():
            tables[path.stem] = pd.read_parquet(path, columns=columns), pd.read_parquet(target, columns=columns)
    labels, total, kept = port_labels(tables)
    print(f"разметка толпы: перенесено {kept} строк из {total}, интервалов {len(labels)}")
    same, different, lost = [], [], []
    for source, target in ((SAME_PERSON, same), (DIFFERENT_PEOPLE, different)):
        for a, b, text in source:
            new_a, new_b = port_point(tables, a), port_point(tables, b)
            if new_a is None or new_b is None:
                lost.append(text)
            else:
                target.append((new_a, new_b, text))
    print(f"точки проверки: {len(same)} и {len(different)}, потеряно: {lost or 'нет'}")
    if not args.write:
        return
    text = TRUTH.read_text(encoding="utf-8")
    block = lambda name, items: f"{name} = [\n" + "".join(f"    {item!r},\n" for item in items) + "]\n"
    def replace(text: str, name: str, items: list) -> str:
        start = text.index(f"{name} = [")
        end = text.index("]\n", start) + 2
        return text[:start] + block(name, items) + text[end:]
    for name, items in (("LABELS", labels), ("SAME_PERSON", same), ("DIFFERENT_PEOPLE", different)):
        text = replace(text, name, items)
    TRUTH.write_text(text, encoding="utf-8")
    print("записано в", TRUTH)


if __name__ == "__main__":
    main()
