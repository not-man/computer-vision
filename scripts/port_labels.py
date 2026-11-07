"""Переносит ручную разметку на новый прогон детектора: номера трекера меняются, а рамки тех же людей остаются на месте.

Строки сопоставляются в общих часах покадрово и взаимно однозначно. --write заменяет
интервалы в evaluation/fragment_truth.py только после ручной проверки нового прогона.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import linear_sum_assignment

from evaluation.fragment_truth import DIFFERENT_PEOPLE, LABELS, SAME_PERSON, TIME_TOLERANCE_S, label_rows

BOX = ["x1", "y1", "x2", "y2"]
MATCH_IOU = 0.5
NEAR_S = (2.0, 5.0, 10.0)  # точка проверки может попасть в кадр без рамки: ищем всё дальше по времени
TRUTH = Path(__file__).resolve().parents[1] / "evaluation" / "fragment_truth.py"
CLOCKS = TRUTH.parents[1] / "configs" / "clocks.yaml"


def clocked_table(frame: pd.DataFrame, camera: str, clocks: dict) -> pd.DataFrame:
    """Использует общие часы проекта, а не метки времени из файла детектора."""
    result = frame.copy()
    clock = clocks[camera]
    result["t_s"] = clock["clock_at_frame_0"] + result.frame / clock["fps"]
    result["camera"] = camera
    return result


def read_tables(old_dir: Path, new_dir: Path) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    clocks = yaml.safe_load(CLOCKS.read_text(encoding="utf-8"))
    columns = ["frame", "raw_track", *BOX]
    tables = {}
    for camera in clocks:
        old_path, new_path = old_dir / f"{camera}.parquet", new_dir / f"{camera}.parquet"
        if not old_path.exists() or not new_path.exists():
            raise FileNotFoundError(f"Для сравнения нужна камера {camera} в обоих прогонах")
        tables[camera] = (
            clocked_table(pd.read_parquet(old_path, columns=columns), camera, clocks),
            clocked_table(pd.read_parquet(new_path, columns=columns), camera, clocks),
        )
    return tables


def transfer_truth_rows(tables: dict[str, tuple[pd.DataFrame, pd.DataFrame]]) -> pd.DataFrame:
    """Каждую размеченную исходную рамку сопоставляет одной рамке нового прогона в том же кадре."""
    records = []
    for camera, (old, new) in tables.items():
        if old.duplicated(["frame", "raw_track"]).any() or new.duplicated(["frame", "raw_track"]).any():
            raise ValueError(f"Повторные рамки одного номера в кадре: {camera}")
        marked = old.assign(person=label_rows(old))
        labeled_frames = marked.loc[marked.person.notna(), "frame"].unique()
        targets = {frame: rows for frame, rows in new[new.frame.isin(labeled_frames)].groupby("frame")}
        for frame, rows in marked[marked.frame.isin(labeled_frames)].groupby("frame"):
            labeled = rows[rows.person.notna()]
            if labeled.empty:
                continue
            matched = {}
            candidates = targets.get(frame)
            if candidates is not None and not candidates.empty:
                overlap = box_iou(rows[BOX].to_numpy(float), candidates[BOX].to_numpy(float))
                source_index, target_index = linear_sum_assignment(-overlap)
                for source, target in zip(source_index, target_index):
                    if overlap[source, target] >= MATCH_IOU:
                        matched[rows.index[source]] = (int(candidates.iloc[target].raw_track), float(overlap[source, target]))
            for index, row in labeled.iterrows():
                target = matched.get(index)
                records.append({"camera": camera, "frame": int(frame), "raw_track": int(row.raw_track),
                                "person": row.person, "new_raw_track": target[0] if target else pd.NA,
                                "match_iou": target[1] if target else 0.0})
    return pd.DataFrame.from_records(records)


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
        source, target = linear_sum_assignment(-iou)
        good = iou[source, target] >= MATCH_IOU
        numbers.loc[part.index[source[good]]] = other.raw_track.to_numpy()[target[good]]
    return numbers


def port_labels(tables: dict[str, tuple[pd.DataFrame, pd.DataFrame]]) -> tuple[list[tuple], int, int]:
    ported, total, kept = [], 0, 0
    for camera, raw_track, start, end, person in LABELS:
        old, new = tables[camera]
        rows = old[(old.raw_track == raw_track) & (old.t_s >= start - TIME_TOLERANCE_S) & (old.t_s <= end + TIME_TOLERANCE_S)].sort_values("t_s")
        all_rows = old[old.frame.isin(rows.frame)]
        numbers = matched_numbers(all_rows, new).reindex(rows.index)
        total += len(rows)
        kept += int(numbers.notna().sum())
        matched = rows[numbers.notna()].copy().sort_values("frame")
        matched["new_raw_track"] = numbers.loc[matched.index].astype(int)
        # Не накрываем одним интервалом пропавшие кадры или временную смену номера.
        boundary = matched.frame.diff().fillna(0).gt(3) | matched.new_raw_track.ne(matched.new_raw_track.shift())
        for _, contiguous in matched.groupby(boundary.cumsum()):
            ported.append((camera, int(contiguous.new_raw_track.iloc[0]), round(float(contiguous.t_s.min()), 3),
                           round(float(contiguous.t_s.max()), 3), person))
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
        all_rows = old[old.frame.isin(rows.frame)]
        numbers = matched_numbers(all_rows, new).reindex(rows.index)
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

    tables = read_tables(args.old, args.new)
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
