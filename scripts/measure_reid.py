"""Пороги ReID по самой записи. Свои: уверенные передачи по геометрии. Чужие: видны одновременно и далеко друг от друга."""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np

from mct.geometry import load_cameras
from mct.plan import Plan
from mct.tracklets import build_tracklets, load_perception
from mct.trinity import Tracklet, link_tracklets

MIN_CONTACT_S = 2.0
MAX_LINK_M = 1.0
FAR_M = 5.0


def unit(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def far_apart(a: Tracklet, b: Tracklet) -> bool:
    start, end = max(a.start, b.start), min(a.end, b.end)
    if end - start < 1.0:
        return False
    times = np.arange(start, end, 0.5)
    return float(np.median(np.hypot(*(a.at(times)[0] - b.at(times)[0]).T))) >= FAR_M


def show(name: str, values: list[float]) -> None:
    print(f"{name:34s} n={len(values):4d}  p5 {np.percentile(values, 5):+.2f}  p25 {np.percentile(values, 25):+.2f}  "
          f"p50 {np.percentile(values, 50):+.2f}  p95 {np.percentile(values, 95):+.2f}  p99 {np.percentile(values, 99):+.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("perception", type=Path)
    args = parser.parse_args()

    observations, embeddings = load_perception(args.perception, load_cameras(), Plan.load())
    tracklets = build_tracklets(observations, embeddings)[0]
    result = link_tracklets(tracklets, zones=None)
    by_id = {x.id: x for x in tracklets if x.embedding is not None}

    same = [
        float(by_id[link.a].embedding @ by_id[link.b].embedding)
        for track in result.tracks for link in track.links
        if link.tier == "sustained" and link.confidence == "green" and link.g_m < MAX_LINK_M and link.contact_s >= MIN_CONTACT_S
        and link.a in by_id and link.b in by_id
    ]
    different = [
        float(a.embedding @ b.embedding) for a, b in combinations(by_id.values(), 2) if a.camera != b.camera and far_apart(a, b)
    ]
    show("один человек, разные камеры", same)
    show("разные люди, разные камеры", different)
    for threshold in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4):
        print(f"  порог {threshold:.2f}: свои {np.mean(np.array(same) >= threshold):.2f}, чужие {np.mean(np.array(different) >= threshold):.3f}")

    groups = {t.track_id: [by_id[i].embedding for i in t.tracklets if i in by_id] for t in result.tracks}
    means = {key: unit(np.mean(vectors, axis=0)) for key, vectors in groups.items() if vectors}
    same_tracks = [float(unit(np.mean(v[::2], axis=0)) @ unit(np.mean(v[1::2], axis=0))) for v in groups.values() if len(v) >= 4]
    simultaneous = [
        float(means[a.track_id] @ means[b.track_id]) for a, b in combinations(result.tracks, 2)
        if a.track_id in means and b.track_id in means and min(a.t_end, b.t_end) - max(a.t_start, b.t_start) > 1.0
    ]
    show("трек: половины одного человека", same_tracks)
    show("треки, идущие одновременно", simultaneous)
    for threshold in (0.4, 0.5, 0.6, 0.7):
        print(f"  порог {threshold:.2f}: свои {np.mean(np.array(same_tracks) >= threshold):.2f}, чужие {np.mean(np.array(simultaneous) >= threshold):.3f}")


if __name__ == "__main__":
    main()
