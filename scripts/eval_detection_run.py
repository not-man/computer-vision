"""Оценка другого прогона детектора или трекера: склейка, толпа и контрольные пары. Папка как artifacts/perception.

Разметка переносится на новый прогон по совпадению рамок прямо в памяти, файл разметки не меняется. Нужны <камера>.parquet, <камера>_emb.npz и <камера>_emb_dense.npz.
"""
import argparse
from pathlib import Path

import pandas as pd
import yaml

import evaluation.fragment_truth as truth
from mct import trinity as tr
from mct.geometry import load_cameras
from mct.plan import Plan
from mct.tracklets import build_tracklets, filter_config, load_perception
from port_labels import BOX, port_labels, port_point


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--base", type=Path, default=Path("artifacts/perception"))
    args = parser.parse_args()

    columns = ["frame", "t_s", "raw_track", *BOX]
    tables = {path.stem: (pd.read_parquet(path, columns=columns), pd.read_parquet(args.run / path.name, columns=columns))
              for path in sorted(args.base.glob("camera_new_?.parquet")) if (args.run / path.name).exists()}
    labels, total, kept = port_labels(tables)
    print(f"разметка толпы: перенесено {kept} строк из {total}")
    same, different, lost = [], [], []
    for source, target in ((truth.SAME_PERSON, same), (truth.DIFFERENT_PEOPLE, different)):
        for a, b, text in source:
            new_a, new_b = port_point(tables, a), port_point(tables, b)
            if new_a is None or new_b is None:
                lost.append(text)
            else:
                target.append((new_a, new_b, text))
    print("контрольных пар", len(same) + len(different), "потеряно", lost or "нет")
    truth.LABELS, truth.SAME_PERSON, truth.DIFFERENT_PEOPLE = labels, same, different

    plan = Plan.load()
    observations, embeddings = load_perception(args.run, load_cameras(), plan)
    extra = [tr.Zone(**zone) for zone in (yaml.safe_load(Path("configs/zones.yaml").read_text(encoding="utf-8")) or {}).get("zones") or []]
    zones = tr.ZoneMap(plan.door_zones() + extra, strict_blind=False)
    tracklets, assigned, dropped = build_tracklets(observations, embeddings, filter_config())
    result = tr.link_tracklets(tracklets, tr.TrinityConfig(), zones)
    owner = {piece: track.track_id for track in result.tracks for piece in track.tracklets}
    numbers = truth.score(assigned, assigned.tracklet.map(owner))
    errors = truth.check_pairs(assigned, assigned.tracklet.map(owner))
    print("наблюдений", len(observations), "кусков", len(tracklets), "отброшено", len(dropped))
    print("треков", len(result.tracks), {key: value for key, value in numbers.items() if key != "по_людям"})
    print(numbers["по_людям"])
    print("пары верно", len(same) + len(different) - sum(map(len, errors.values())), "из", len(same) + len(different), errors)
    links = pd.Series([link.tier for track in result.tracks for link in track.links]).value_counts().to_dict()
    print("связи", links, "спорных", sum(link.confidence == "yellow" for track in result.tracks for link in track.links),
          "нарушений", len(tr.check_invariants(result, tracklets, tr.TrinityConfig())))


if __name__ == "__main__":
    main()
