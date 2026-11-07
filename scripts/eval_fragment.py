"""Качество на размеченном вручную фрагменте: соответствие человек-трек, чистота и полнота. Разметка в evaluation.fragment_truth."""
import argparse
from pathlib import Path

import yaml

from mct import trinity as tr
from evaluation.fragment_truth import DIFFERENT_PEOPLE, SAME_PERSON, check_pairs, score
from mct.geometry import load_cameras
from mct.plan import Plan
from mct.tracklets import build_tracklets, filter_config, load_perception


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--perception", type=Path, default=Path("artifacts/perception"))
    args = parser.parse_args()
    plan = Plan.load()
    observations, embeddings = load_perception(args.perception, load_cameras(), plan)
    extra = [tr.Zone(**zone) for zone in (yaml.safe_load(Path("configs/zones.yaml").read_text(encoding="utf-8")) or {}).get("zones") or []]
    zones = tr.ZoneMap(plan.door_zones() + extra, strict_blind=False)
    tracklets, assigned, _ = build_tracklets(observations, embeddings, filter_config())
    result = tr.link_tracklets(tracklets, tr.TrinityConfig(), zones)
    owner = {i: track.track_id for track in result.tracks for i in track.tracklets}
    numbers = score(assigned, assigned.tracklet.map(owner))
    print("треков всего", len(result.tracks), {k: v for k, v in numbers.items() if k != "по_людям"})
    print(numbers["по_людям"])
    errors = check_pairs(assigned, assigned.tracklet.map(owner))
    print(f"проверка глазами: {len(SAME_PERSON) + len(DIFFERENT_PEOPLE) - sum(map(len, errors.values()))} из {len(SAME_PERSON) + len(DIFFERENT_PEOPLE)} пар верно")
    for name, items in errors.items():
        print(" ", name, items)


if __name__ == "__main__":
    main()
