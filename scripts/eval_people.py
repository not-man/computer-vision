"""Счёт по таблице людей: сколько треков должно быть по правилу минуты и какие лишние.

Таблица evaluation/people_truth.csv составлена по снимкам из исходных кадров (scripts/draw_track_strips.py). Треки одного человека
с паузой не дольше минуты по правилу должны быть одним треком. Паузы около минуты печатаются отдельно: их стоит досмотреть по видео.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from mct.video import clock_text

WINDOW_S = 60.0
BORDER_S = (50.0, 80.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", type=Path, default=Path("artifacts/tracks/tracks.json"))
    parser.add_argument("--people", type=Path, default=Path("evaluation/people_truth.csv"))
    args = parser.parse_args()

    tracks = {track["track_id"]: track for track in json.loads(args.tracks.read_text(encoding="utf-8"))["tracks"]}
    people = pd.read_csv(args.people)
    missing = sorted(set(tracks) - set(people.track))
    if missing:
        raise SystemExit(f"В таблице людей нет треков {missing}: таблица устарела после пересчёта")
    extra, border = [], []
    for person, part in people.groupby("person"):
        own = sorted((tracks[track] for track in part.track), key=lambda track: track["t_start_ms"])
        for before, after in zip(own, own[1:]):
            pause = (after["t_start_ms"] - before["t_end_ms"]) / 1000
            if pause <= WINDOW_S:
                extra.append((person, before["track_id"], after["track_id"], pause))
            if BORDER_S[0] <= pause <= BORDER_S[1]:
                border.append((person, before["track_id"], after["track_id"], pause))
    print(f"треков {len(tracks)}, людей в таблице {people.person.nunique()}, уверенных подписей {int((people.confidence == 'уверенно').sum())}")
    print(f"по правилу минуты должно быть {len(tracks) - len(extra)}")
    for person, a, b, pause in extra:
        print(f"  лишний: {person} #{a} и #{b}, пауза {pause:.1f} с")
    for person, a, b, pause in border:
        print(f"  у границы минуты: {person} #{a} {clock_text(tracks[a]['t_end_ms'] / 1000)[3:10]} и #{b} {clock_text(tracks[b]['t_start_ms'] / 1000)[3:10]}, пауза {pause:.1f} с")
    mixed = people[people.note.str.startswith("смешение")]
    for row in mixed.itertuples():
        print(f"  смешение: #{row.track} {row.note}")


if __name__ == "__main__":
    main()
