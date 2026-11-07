"""Сквозные треки из результатов перцепции: tracks.json, tracks.csv, все проверенные пары и отброшенные детекции."""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # матрицы мелкие: потоки только мешают, когда считают все камеры сразу

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from mct.fragment_truth import DIFFERENT_PEOPLE, SAME_PERSON, check_pairs, score
from mct.geometry import load_cameras
from mct.plan import Plan
from mct.tracklets import build_tracklets, filter_config, load_perception
from mct.trinity import LinkResult, Tracklet, TrinityConfig, Zone, ZoneMap, check_invariants, link_tracklets
from mct.video import clock_text

CONFIGS = [Path("configs/cameras.yaml"), Path("configs/clocks.yaml"), Path("configs/plan_features.yaml"), Path("configs/zones.yaml")]
PATH_STEP_S = 0.5


def plan_path(track_tracklets: list[Tracklet], start: float, end: float) -> list[list[float]]:
    """Путь на плане: в каждый момент среднее по видящим камерам с весом их точности."""
    path = []
    for t in np.arange(start, end + 1e-9, PATH_STEP_S):
        positions, weights = [], []
        for tracklet in track_tracklets:
            if tracklet.start - 1e-9 <= t <= tracklet.end + 1e-9:
                xy, cov, known = tracklet.at(np.array([t]))
                if known[0]:
                    positions.append(xy[0])
                    weights.append(1.0 / float(np.trace(cov[0]) + 1e-3))
        if positions:
            mean = np.average(positions, axis=0, weights=weights)
            path.append([round(float(mean[0]), 2), round(float(mean[1]), 2), int(round(t * 1000))])
    return path


def measured_quality(result: LinkResult, assigned: pd.DataFrame) -> dict:
    """Что измерено вручную размеченным: толпа 22 с и пары из просмотра всех треков глазами."""
    owner = {i: track.track_id for track in result.tracks for i in track.tracklets}
    track_of_row = assigned.tracklet.map(owner)
    crowd = score(assigned, track_of_row)
    errors = check_pairs(assigned, track_of_row)
    total = len(SAME_PERSON) + len(DIFFERENT_PEOPLE)
    return {
        "crowd": {"match": crowd["соответствие"], "purity": crowd["чистота"], "people": crowd["людей"], "tracks": crowd["треков"]},
        "pairs": {"right": total - sum(map(len, errors.values())), "total": total, "errors": errors},
    }


def to_json(result: LinkResult, tracklets: list[Tracklet], config_hash: str, embedder: str) -> dict:
    by_id = {x.id: x for x in tracklets}
    tracks = []
    for track in result.tracks:
        members = [by_id[i] for i in track.tracklets]
        tracks.append({
            "track_id": track.track_id,
            "t_start_ms": int(round(track.t_start * 1000)),
            "t_end_ms": int(round(track.t_end * 1000)),
            "start": clock_text(track.t_start),
            "end": clock_text(track.t_end),
            "cameras": track.cameras,
            "close_zone": track.close_zone,
            "tracklets": track.tracklets,
            "segments": [
                {"kind": s.kind, "camera_id": s.camera, "t_start_ms": int(round(s.t_start * 1000)), "t_end_ms": int(round(s.t_end * 1000))}
                for s in track.segments
            ],
            "links": [
                {"a": link.a, "b": link.b, "camera_a": by_id[link.a].camera, "camera_b": by_id[link.b].camera,
                 "tier": link.tier, "t_ms": int(round(link.t * 1000)), "g_m": round(link.g_m, 2),
                 "contact_s": round(link.contact_s, 2), "r": None if link.r is None else round(link.r, 2),
                 "score": round(link.score, 3), "confidence": link.confidence}
                for link in track.links
            ],
            "plan_path": plan_path(members, track.t_start, track.t_end),
            "best_crop": None,
            "clip": None,
            "possibly_same_person_as": [
                {"track_id": h.track_id, "r": round(h.r, 2), "absent_s": round(h.absent_s, 1)} for h in track.hints
            ],
        })
    return {
        "run_id": config_hash[:6],
        "config_hash": f"sha256:{config_hash}",
        "embedder": embedder,
        "time_origin": "секунды и миллисекунды от 17:00:00 по часам камер",
        "total_tracks": len(tracks),
        "tracks": tracks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("perception", type=Path)
    parser.add_argument("--out", type=Path, default=Path("artifacts/tracks"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    plan = Plan.load()
    cameras = load_cameras()
    observations, embeddings = load_perception(args.perception, cameras, plan)
    tracklets, assigned, dropped = build_tracklets(observations, embeddings, filter_config())

    zones_config = yaml.safe_load(Path("configs/zones.yaml").read_text(encoding="utf-8")) or {}
    extra = [Zone(**zone) for zone in zones_config.get("zones") or []]
    zones = ZoneMap(plan.door_zones() + extra, strict_blind=bool(zones_config.get("strict_blind", False)))
    config = TrinityConfig()
    result = link_tracklets(tracklets, config, zones)
    problems = check_invariants(result, tracklets, config)

    digest = hashlib.sha256()
    for path in CONFIGS:
        # Windows Git can change YAML line endings without changing settings.
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    summaries = sorted(args.perception.glob("*_summary.json"))
    embedder = json.loads(summaries[0].read_text(encoding="utf-8")).get("embedder", "") if summaries else ""
    document = to_json(result, tracklets, digest.hexdigest(), embedder)
    document["quality"] = measured_quality(result, assigned)
    (args.out / "tracks.json").write_text(json.dumps(document, ensure_ascii=False, indent=1), encoding="utf-8")
    pd.DataFrame([
        {"track_id": t.track_id, "start": clock_text(t.t_start), "end": clock_text(t.t_end), "cameras": ", ".join(t.cameras)}
        for t in result.tracks
    ]).to_csv(args.out / "tracks.csv", index=False)
    pd.DataFrame(result.checks).to_parquet(args.out / "links.parquet", index=False)
    dropped.to_csv(args.out / "dropped.csv", index=False)
    assigned.drop(columns=["cov_xx", "cov_xy", "cov_yy"]).to_parquet(args.out / "observations.parquet", index=False)

    tiers = pd.Series([link.tier for track in result.tracks for link in track.links]).value_counts().to_dict()
    yellow = sum(link.confidence == "yellow" for track in result.tracks for link in track.links)
    print(f"наблюдений {len(observations)}, треклетов {len(tracklets)}, отброшено {len(dropped)}")
    if len(dropped):
        print(dropped.reason.value_counts().to_string())
    print(f"треков {len(result.tracks)}, отброшено коротких {len(result.dropped)} треклетов, связи {tiers}, жёлтых {yellow}")
    print(f"подсказок Призрака {sum(len(t.hints) for t in result.tracks)}")
    print("инварианты:", problems if problems else "нарушений нет")
    for track in result.tracks:
        print(f"  #{track.track_id}: {clock_text(track.t_start)}–{clock_text(track.t_end)} {','.join(c[-1] for c in track.cameras)} "
              f"треклеты {track.tracklets} закрыт {track.close_zone} "
              + " ".join(f"{link.tier[0].upper()}:{link.g_m:.1f}м" for link in track.links))


if __name__ == "__main__":
    main()
