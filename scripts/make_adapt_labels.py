"""Подписи для дообучения внешности на самой записи: artifacts/adapt/labels.parquet и different.csv. Ручной разметки здесь нет.

Свои: куски одной цепочки связей перекрытия без спорных, то есть две камеры долго видели человека в одном месте.
Чужие: люди видны одновременно и далеко друг от друга на плане или в непересекающихся рамках одной камеры.
Про остальные пары отношение неизвестно. Размеченная толпа и треки с известными ошибками в подписи не входят.
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from mct.summary import KNOWN_ERRORS

CROWD_S = (2720.0, 2752.0)  # 17:45:20–17:45:52: размеченная толпа с запасом, по ней потом проверяем
SUSPECT_TRACKS = {error.track_id for error in KNOWN_ERRORS} | {25}  # 25: у него хвост на соседе, см. пример с женщиной в полосатом
MIN_TOGETHER_S = 1.0
APART_M = 4.0
SAMPLE_S = 0.2
MIN_SHARED_FRAMES = 10
DISJOINT_SHARE = 0.9


def chains_of(tracks: list[dict]) -> dict[int, int]:
    parent = {piece: piece for track in tracks for piece in track["tracklets"]}

    def find(piece: int) -> int:
        while parent[piece] != piece:
            parent[piece] = parent[parent[piece]]
            piece = parent[piece]
        return piece

    for track in tracks:
        for link in track["links"]:
            if link["tier"] == "sustained" and link["confidence"] != "yellow":
                parent[find(link["a"])] = find(link["b"])
    return {piece: find(piece) for piece in parent}


def boxes_disjoint(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """Два куска одной камеры почти всё общее время в непересекающихся рамках: это два объекта."""
    shared = a.merge(b, on="frame", suffixes=("_a", "_b"))
    if len(shared) < MIN_SHARED_FRAMES:
        return False
    apart = (shared.x2_a <= shared.x1_b) | (shared.x2_b <= shared.x1_a) | (shared.y2_a <= shared.y1_b) | (shared.y2_b <= shared.y1_a)
    return float(apart.mean()) >= DISJOINT_SHARE


def different_pieces(table: pd.DataFrame) -> list[tuple[int, int]]:
    pieces = {piece: part.sort_values("t_s") for piece, part in table.groupby("tracklet")}
    found = []
    for (i, a), (j, b) in combinations(pieces.items(), 2):
        low, high = max(a.t_s.iloc[0], b.t_s.iloc[0]), min(a.t_s.iloc[-1], b.t_s.iloc[-1])
        if high - low < MIN_TOGETHER_S:
            continue
        times = np.arange(low, high, SAMPLE_S)
        gaps = np.hypot(np.interp(times, a.t_s, a.X) - np.interp(times, b.t_s, b.X), np.interp(times, a.t_s, a.Y) - np.interp(times, b.t_s, b.Y))
        if gaps.min() >= APART_M or (a.camera.iloc[0] == b.camera.iloc[0] and boxes_disjoint(a, b)):
            found.append((int(i), int(j)))
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", type=Path, default=Path("artifacts/tracks"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/adapt"))
    args = parser.parse_args()

    tracks = json.loads((args.tracks / "tracks.json").read_text(encoding="utf-8"))["tracks"]
    table = pd.read_parquet(args.tracks / "observations.parquet", columns=["camera", "frame", "raw_track", "t_s", "tracklet", "touches_edge", "X", "Y", "x1", "y1", "x2", "y2"])
    owner = {piece: track["track_id"] for track in tracks for piece in track["tracklets"]}
    chain = chains_of(tracks)
    table = table[table.tracklet.isin(owner)]

    different = set()
    for i, j in different_pieces(table):
        if owner[i] != owner[j] and owner[i] not in SUSPECT_TRACKS and owner[j] not in SUSPECT_TRACKS:
            different.add(tuple(sorted((chain[i], chain[j]))))

    kept = table[~table.touches_edge & ~table.tracklet.map(owner).isin(SUSPECT_TRACKS) & ~table.t_s.between(*CROWD_S)]
    labels = kept[["camera", "frame", "raw_track", "t_s"]].assign(chain=kept.tracklet.map(chain).astype(int))
    args.out.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(args.out / "labels.parquet", index=False)
    pd.DataFrame(sorted(different), columns=["a", "b"]).to_csv(args.out / "different.csv", index=False)
    cameras = labels.groupby("chain").camera.nunique()
    print(f"подписей {len(labels)}, цепочек {len(cameras)}, из них на нескольких камерах {(cameras > 1).sum()}, заведомо чужих пар цепочек {len(different)}")


if __name__ == "__main__":
    main()
