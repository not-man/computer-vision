"""Чем отличаются два результата склейки: какие треки слились, разделились и у каких сменилась причина закрытия.

Номера кусков и треков при пересчёте меняются, поэтому сравнение идёт по наблюдениям: камера, кадр, номер трекера.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from mct.video import clock_text

KEYS = ["camera", "frame", "raw_track"]
MIN_ROWS = 5  # меньше этого считаем дребезгом на границе


def load(folder: Path) -> tuple[dict[int, dict], pd.DataFrame]:
    tracks = {track["track_id"]: track for track in json.loads((folder / "tracks.json").read_text(encoding="utf-8"))["tracks"]}
    owner = {piece: track_id for track_id, track in tracks.items() for piece in track["tracklets"]}
    rows = pd.read_parquet(folder / "observations.parquet", columns=[*KEYS, "tracklet"])
    return tracks, rows.assign(track=rows.tracklet.map(owner)).dropna(subset=["track"]).drop(columns="tracklet").drop_duplicates(KEYS)


def describe(track: dict) -> str:
    return f"#{track['track_id']} {clock_text(track['t_start_ms'] / 1000)[3:8]}–{clock_text(track['t_end_ms'] / 1000)[3:8]}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("new", type=Path)
    parser.add_argument("--old", type=Path, default=Path("artifacts/tracks"))
    args = parser.parse_args()

    old_tracks, old_rows = load(args.old)
    new_tracks, new_rows = load(args.new)
    pairs = old_rows.merge(new_rows, on=KEYS, suffixes=("_old", "_new"))
    sizes = pairs.groupby(["track_old", "track_new"]).size()
    sizes = sizes[sizes >= MIN_ROWS].reset_index(name="rows")
    print(f"треков было {len(old_tracks)}, стало {len(new_tracks)}; наблюдений только в старом {len(old_rows) - len(pairs)}, только в новом {len(new_rows) - len(pairs)}")
    for new_id, part in sizes.groupby("track_new"):
        if len(part) > 1:
            print("слились в", describe(new_tracks[int(new_id)]), "из", ", ".join(f"{describe(old_tracks[int(row.track_old)])} ({row.rows} набл.)" for row in part.itertuples()))
    for old_id, part in sizes.groupby("track_old"):
        if len(part) > 1:
            print("разделился", describe(old_tracks[int(old_id)]), "на", ", ".join(f"{describe(new_tracks[int(row.track_new)])} ({row.rows} набл.)" for row in part.itertuples()))
    single_old, single_new = sizes.groupby("track_old").size(), sizes.groupby("track_new").size()
    for row in sizes.itertuples():
        if single_old[row.track_old] == 1 and single_new[row.track_new] == 1:
            before, after = old_tracks[int(row.track_old)], new_tracks[int(row.track_new)]
            if before["close_zone"] != after["close_zone"]:
                print("закрытие", describe(after), before["close_zone"], "->", after["close_zone"])
    lost = set(old_tracks) - {int(value) for value in sizes.track_old}
    born = set(new_tracks) - {int(value) for value in sizes.track_new}
    if lost:
        print("пропали:", ", ".join(describe(old_tracks[value]) for value in sorted(lost)))
    if born:
        print("появились:", ", ".join(describe(new_tracks[value]) for value in sorted(born)))


if __name__ == "__main__":
    main()
