"""Поправки к clock_at_frame_0 по людям на нескольких камерах. Запускать после link_tracks.py.

Поправку вносят в configs/clocks.yaml вручную и пересчитывают треки: номера кадров от неё не зависят.
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import pandas as pd
import yaml

from mct.sync import PairLag, camera_offsets, pair_lag


def collect_lags(tracks_dir: Path) -> list[PairLag]:
    observations = pd.read_parquet(tracks_dir / "observations.parquet")
    observations = observations[observations.source == "foot"]
    paths = {tracklet: group.sort_values("t_s") for tracklet, group in observations.groupby("tracklet")}
    result = json.loads((tracks_dir / "tracks.json").read_text(encoding="utf-8"))

    lags = []
    for track in result["tracks"]:
        present = [tracklet for tracklet in track["tracklets"] if tracklet in paths]
        for first, second in combinations(present, 2):
            a, b = sorted((first, second), key=lambda tracklet: paths[tracklet].camera.iloc[0])
            camera_a, camera_b = paths[a].camera.iloc[0], paths[b].camera.iloc[0]
            if camera_a == camera_b:
                continue
            found = pair_lag(
                camera_a, paths[a].t_s.to_numpy(), paths[a][["X", "Y"]].to_numpy(),
                camera_b, paths[b].t_s.to_numpy(), paths[b][["X", "Y"]].to_numpy(),
            )
            if found is not None:
                lags.append(found)
    return lags


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", type=Path, default=Path("artifacts/tracks"))
    args = parser.parse_args()

    clocks = yaml.safe_load(Path("configs/clocks.yaml").read_text(encoding="utf-8"))
    lags = collect_lags(args.tracks)
    table = pd.DataFrame([vars(item) for item in lags])
    table["pair"] = table.camera_a.str[-1] + "-" + table.camera_b.str[-1]
    print(table.groupby(["pair", "direction"]).agg(
        n=("lag", "size"), lag=("lag", "median"), q25=("lag", lambda s: s.quantile(0.25)), q75=("lag", lambda s: s.quantile(0.75)),
        distance=("distance", "median"), distance_at_zero=("distance_at_zero", "median"),
    ).round(2).to_string())

    offsets, residuals = camera_offsets(lags, list(clocks))
    print("пар:", len(lags), " остатки:", {f"{a[-1]}-{b[-1]}": round(value, 2) for (a, b), value in residuals.items()})
    print("время камеры опережает общее, с:", {name[-1]: round(value, 2) for name, value in offsets.items()})
    print("новый clock_at_frame_0:", {name: round(clocks[name]["clock_at_frame_0"] - value, 3) for name, value in offsets.items()})


if __name__ == "__main__":
    main()
