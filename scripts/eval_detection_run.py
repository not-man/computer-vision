"""Оценка другого прогона детектора или трекера: склейка, толпа и контрольные пары. Папка как artifacts/perception.

Разметка переносится по рамкам в том же кадре с общими часами, файл разметки не меняется.
Пропуски остаются в знаменателе. Нужны <камера>.parquet и <камера>_emb.npz;
<камера>_emb_dense.npz используется, если есть.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

import evaluation.fragment_truth as truth
from mct import trinity as tr
from mct.geometry import load_cameras
from mct.plan import Plan
from mct.tracklets import build_tracklets, filter_config, link_with_tail_cuts, load_perception
from scripts.port_labels import port_point, read_tables, transfer_truth_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--base", type=Path, default=Path("artifacts/perception"))
    parser.add_argument("--base-tracks", type=Path, default=Path("artifacts/tracks"))
    args = parser.parse_args()

    tables = read_tables(args.base, args.run)
    marked = transfer_truth_rows(tables)
    total, kept = len(marked), int(marked.new_raw_track.notna().sum())
    print(f"разметка толпы по общим часам: перенесено {kept} строк из {total}")
    same, different, lost = [], [], []
    for source, target in ((truth.SAME_PERSON, same), (truth.DIFFERENT_PEOPLE, different)):
        for a, b, text in source:
            new_a, new_b = port_point(tables, a), port_point(tables, b)
            if new_a is None or new_b is None:
                lost.append(text)
            else:
                target.append((new_a, new_b, text))
    print("контрольных пар", len(same) + len(different), "потеряно", lost or "нет")
    truth.SAME_PERSON, truth.DIFFERENT_PEOPLE = same, different

    plan = Plan.load()
    observations, embeddings = load_perception(args.run, load_cameras(), plan)
    extra = [tr.Zone(**zone) for zone in (yaml.safe_load(Path("configs/zones.yaml").read_text(encoding="utf-8")) or {}).get("zones") or []]
    zones = tr.ZoneMap(plan.door_zones() + extra, strict_blind=False)
    tracklets, assigned, dropped = build_tracklets(observations, embeddings, filter_config())
    result, tracklets, assigned = link_with_tail_cuts(tracklets, assigned, embeddings, tr.TrinityConfig(), zones)
    owner = {piece: track.track_id for track in result.tracks for piece in track.tracklets}
    baseline = pd.read_parquet(args.base_tracks / "observations.parquet", columns=["camera", "frame", "raw_track", "tracklet"])
    baseline_tracks = json.loads((args.base_tracks / "tracks.json").read_text(encoding="utf-8"))["tracks"]
    baseline_owner = {piece: track["track_id"] for track in baseline_tracks for piece in track["tracklets"]}
    base_score = truth.score_fixed_truth(marked.person, truth.track_on_truth(marked, baseline, baseline.tracklet.map(baseline_owner), target="raw_track"))
    predicted = truth.track_on_truth(marked, assigned, assigned.tracklet.map(owner), target="new_raw_track")
    numbers = truth.score_fixed_truth(marked.person, predicted)
    errors = truth.check_pairs(assigned, assigned.tracklet.map(owner))
    print("наблюдений", len(observations), "кусков", len(tracklets), "отброшено", len(dropped))
    print("основной вариант на той же разметке", base_score)
    print("треков", len(result.tracks), "новый вариант", numbers)
    labeled_ids = pd.DataFrame({"person": marked.person, "track": predicted})
    print("распределение ID по людям", {
        person: {"без ID" if pd.isna(track) else str(int(track)): int(count)
                 for track, count in group.track.value_counts(dropna=False).items()}
        for person, group in labeled_ids.groupby("person")
    })
    print("пары верно", len(same) + len(different) - sum(map(len, errors.values())), "из", len(same) + len(different), errors)
    links = pd.Series([link.tier for track in result.tracks for link in track.links]).value_counts().to_dict()
    print("связи", links, "спорных", sum(link.confidence == "yellow" for track in result.tracks for link in track.links),
          "нарушений", len(tr.check_invariants(result, tracklets, tr.TrinityConfig())))


if __name__ == "__main__":
    main()
