"""Пары треков с паузой меньше минуты, которые не склеились, по убыванию сходства: кандидаты на упущенный возврат для просмотра глазами."""
import argparse
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from mct import trinity as tr
from mct.geometry import load_cameras
from mct.plan import Plan
from mct.tracklets import build_tracklets, filter_config, link_with_tail_cuts, load_perception
from mct.video import clock_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--perception", type=Path, default=Path("artifacts/perception"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()

    plan = Plan.load()
    observations, embeddings = load_perception(args.perception, load_cameras(), plan)
    tracklets, assigned, _ = build_tracklets(observations, embeddings, filter_config())
    extra = [tr.Zone(**zone) for zone in (yaml.safe_load(Path("configs/zones.yaml").read_text(encoding="utf-8")) or {}).get("zones") or []]
    zones = tr.ZoneMap(plan.door_zones() + extra, strict_blind=False)
    cfg = tr.TrinityConfig()
    result, tracklets, _ = link_with_tail_cuts(tracklets, assigned, embeddings, cfg, zones)
    by_id = {x.id: x for x in tracklets}
    linker = tr._Linker(tracklets, cfg, zones)
    members = {track.track_id: [by_id[i] for i in track.tracklets] for track in result.tracks}
    vectors = {key: tr._members_embedding(value) for key, value in members.items()}
    rows = []
    for a, b in permutations(members, 2):
        end, start = max(x.end for x in members[a]), min(x.start for x in members[b])
        gap = start - end
        if gap <= 0 or gap > cfg.return_window_s or vectors[a] is None or vectors[b] is None:
            continue
        last, head = max(members[a], key=lambda x: x.end), min(members[b], key=lambda x: x.start)
        distance = float(np.hypot(*(last.xy[-1] - head.xy[0])))
        rows.append({"a": a, "b": b, "r": round(float(vectors[a] @ vectors[b]), 2), "gap_s": round(gap, 1), "dist_m": round(distance, 1),
                     "reachable": linker._return_link(members[a], members[b]) is not None, "end": clock_text(end)[3:8], "start": clock_text(start)[3:8],
                     "cam_a": last.camera[-1], "cam_b": head.camera[-1]})
    table = pd.DataFrame(rows).sort_values("r", ascending=False)
    print(table.head(args.top).to_string(index=False))
    if args.out:
        table.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
