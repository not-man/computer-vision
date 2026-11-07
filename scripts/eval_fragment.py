"""Качество на размеченном вручную фрагменте: соответствие человек-трек, чистота и полнота. Разметка в mct.fragment_truth."""
from dataclasses import replace
from pathlib import Path

import yaml

from mct import trinity as tr
from mct.fragment_truth import score
from mct.geometry import load_cameras
from mct.plan import Plan
from mct.tracklets import build_tracklets, filter_config, load_perception


def main() -> None:
    plan = Plan.load()
    observations, embeddings = load_perception(Path("artifacts/perception"), load_cameras(), plan)
    cfg = filter_config()
    extra = [tr.Zone(**zone) for zone in (yaml.safe_load(Path("configs/zones.yaml").read_text(encoding="utf-8")) or {}).get("zones") or []]
    zones = tr.ZoneMap(plan.door_zones() + extra, strict_blind=False)
    # Разметка привязана к номерам кусков до разрезов
    _, base, _ = build_tracklets(observations, embeddings, replace(cfg, jump_max_cross=-1e9, swap_min_gain=1e9))
    tracklets, assigned, _ = build_tracklets(observations, embeddings, cfg)
    result = tr.link_tracklets(tracklets, tr.TrinityConfig(), zones)
    owner = {i: track.track_id for track in result.tracks for i in track.tracklets}
    key = ["camera", "frame", "raw_track"]
    merged = base[key + ["tracklet", "t_s"]].merge(assigned[key + ["tracklet"]], on=key, suffixes=("", "_new"))
    numbers = score(merged, merged.tracklet_new.map(owner))
    print("треков всего", len(result.tracks), {k: v for k, v in numbers.items() if k != "по_людям"})
    print(numbers["по_людям"])


if __name__ == "__main__":
    main()
