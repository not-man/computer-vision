"""Сравнение вариантов внешности без эталона. Вариант: папка с <камера>_emb.npz, ключи и векторы как в scripts/reembed.py.

Свои и чужие пары выбираются только по геометрии и времени, поэтому модель не проверяет сама себя.
Свои: два куска на разных камерах долго идут вместе, рядом нет соперника; куски одной цепочки устойчивых связей с паузой между ними.
Чужие: видны одновременно и всё это время далеко друг от друга. Дополнительно: снимки размеченной толпы и вся склейка на этих векторах.
"""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from mct import trinity as tr
from mct.fragment_truth import DIFFERENT_PEOPLE, SAME_PERSON, check_pairs, label_rows, score
from mct.geometry import load_cameras
from mct.plan import Plan
from mct.tracklets import build_tracklets, filter_config, load_perception

MIN_TOGETHER_S = 2.0
MAX_MEDIAN_M = 1.0
MAX_P90_M = 1.6
RIVAL_MARGIN_M = 1.0
MIN_APART_S = 1.0
MIN_APART_M = 4.0
PAUSE_S = 5.0


def unit(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


def distances(a: tr.Tracklet, b: tr.Tracklet) -> np.ndarray | None:
    low, high = max(a.start, b.start), min(a.end, b.end)
    if high - low < 0.4:
        return None
    times = np.arange(low, high + 1e-6, 0.2)
    return np.hypot(*(a.at(times)[0] - b.at(times)[0]).T)


def geometric_pairs(tracklets: list[tr.Tracklet]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    sure, apart = [], []
    for a, b in combinations(tracklets, 2):
        gaps = distances(a, b)
        if gaps is None:
            continue
        together = min(a.end, b.end) - max(a.start, b.start)
        if together >= MIN_APART_S and gaps.min() >= MIN_APART_M:
            apart.append((a.id, b.id))
        if a.camera == b.camera or together < MIN_TOGETHER_S or np.median(gaps) > MAX_MEDIAN_M or np.quantile(gaps, 0.9) > MAX_P90_M:
            continue
        low, high = max(a.start, b.start), min(a.end, b.end)
        times = np.arange(low, high + 1e-6, 0.2)
        clear = True
        for c in tracklets:
            if c.id in (a.id, b.id) or c.camera not in (a.camera, b.camera) or c.end <= low or c.start >= high:
                continue
            inside = times[(times >= c.start) & (times <= c.end)]
            partner = b if c.camera == a.camera else a
            if len(inside) >= 3 and np.median(np.hypot(*(c.at(inside)[0] - partner.at(inside)[0]).T)) < np.median(gaps) + RIVAL_MARGIN_M:
                clear = False
                break
        if clear:
            sure.append((a.id, b.id))
    return sure, apart


def chain_pairs(result: tr.LinkResult, by_id: dict[int, tr.Tracklet]) -> list[tuple[int, int]]:
    """Куски одной цепочки устойчивых связей на разных камерах с паузой между ними: так выглядит возврат."""
    parent = {i: i for i in by_id}

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    links = [link for track in result.tracks for link in track.links if link.tier == "sustained"]
    for link in links:
        parent[find(link.a)] = find(link.b)
    chains: dict[int, list[int]] = {}
    for i in by_id:
        chains.setdefault(find(i), []).append(i)
    pairs = []
    for chain in chains.values():
        for a, b in combinations(chain, 2):
            first, second = (by_id[a], by_id[b]) if by_id[a].start <= by_id[b].start else (by_id[b], by_id[a])
            if first.camera != second.camera and second.start - first.end >= PAUSE_S:
                pairs.append((a, b))
    return pairs


def piece_vectors(snapshots: dict[str, tuple[np.ndarray, np.ndarray]], assigned: pd.DataFrame) -> dict[int, np.ndarray]:
    """Как в mct.tracklets: вычитаем среднее камеры, нормируем, усредняем по куску."""
    out = {}
    for camera, (keys, vectors) in snapshots.items():
        rows = pd.DataFrame(keys, columns=["frame", "raw_track"])
        mine = assigned[assigned.camera == camera][["frame", "raw_track", "tracklet"]].drop_duplicates(["frame", "raw_track"])
        owners = rows.merge(mine, on=["frame", "raw_track"], how="left").tracklet.to_numpy()
        real = ~np.isnan(owners)
        centered = unit(vectors[real] - vectors[real].mean(axis=0))
        owners = owners[real]
        for owner in np.unique(owners):
            out[int(owner)] = unit(centered[owners == owner].mean(axis=0))
    return out


def crowd_similarity(snapshots: dict[str, tuple[np.ndarray, np.ndarray]], assigned: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Снимки размеченной толпы: сходство своих и чужих пар между разными камерами."""
    parts = []
    for camera, (keys, vectors) in snapshots.items():
        rows = pd.DataFrame(keys, columns=["frame", "raw_track"]).assign(position=lambda t: np.arange(len(t)))
        mine = assigned[assigned.camera == camera][["frame", "raw_track", "t_s", "camera", "tracklet"]].drop_duplicates(["frame", "raw_track"])
        table = rows.merge(mine, on=["frame", "raw_track"], how="inner")
        real = vectors[table.position.to_numpy()]
        mean = real.mean(axis=0)
        table["person"] = label_rows(table)
        chosen = table[table.person.notna()]
        if len(chosen):
            parts.append((unit(vectors[chosen.position.to_numpy()] - mean), chosen.person.to_numpy(), np.full(len(chosen), camera)))
    if not parts:
        return np.empty(0), np.empty(0)
    v = np.vstack([p[0] for p in parts])
    people, cameras = np.concatenate([p[1] for p in parts]), np.concatenate([p[2] for p in parts])
    similarity = v @ v.T
    upper = np.triu(np.ones_like(similarity, bool), 1) & (cameras[:, None] != cameras[None, :])
    same = people[:, None] == people[None, :]
    return similarity[upper & same], similarity[upper & ~same]


def stats(pos: list | np.ndarray, neg: list | np.ndarray) -> str:
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg):
        return "нет пар"
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    auc = (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return f"{auc:.3f}/{(pos >= np.quantile(neg, 0.97)).mean():.2f}"


def load_variant(folder: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out = {}
    for path in sorted(folder.glob("*_emb*.npz")):
        data = np.load(path)
        out[path.name.split("_emb")[0]] = data["keys"], data["vectors"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("variants", type=Path, nargs="*", help="папки с векторами")
    parser.add_argument("--perception", type=Path, default=Path("artifacts/perception"))
    args = parser.parse_args()

    plan = Plan.load()
    observations, embeddings = load_perception(args.perception, load_cameras(), plan)
    tracklets, assigned, _ = build_tracklets(observations, embeddings, filter_config())
    extra = [tr.Zone(**zone) for zone in (yaml.safe_load(Path("configs/zones.yaml").read_text(encoding="utf-8")) or {}).get("zones") or []]
    zones = tr.ZoneMap(plan.door_zones() + extra, strict_blind=False)
    base = tr.link_tracklets(tracklets, tr.TrinityConfig(), zones)
    by_id = {x.id: x for x in tracklets}
    sure, apart = geometric_pairs(tracklets)
    sustained = [(link.a, link.b) for track in base.tracks for link in track.links if link.tier == "sustained"]
    later = chain_pairs(base, by_id)
    other_camera = [p for p in apart if by_id[p[0]].camera != by_id[p[1]].camera]
    print(f"своих строгих {len(sure)}, по устойчивым связям {len(sustained)}, после паузы {len(later)}, чужих {len(other_camera)}")
    print("Первая цифра AUC, вторая: доля своих при 3% чужих. Толпа: снимки; склейка: треков, толпа, пары из проверки глазами")

    variants = {"раз в секунду": {c: v[:2] for c, v in embeddings.items()}}
    if all(len(v) > 2 and v[2] is not None for v in embeddings.values()):
        variants["чаще (как сейчас)"] = {c: v[2] for c, v in embeddings.items()}
    variants.update({folder.name: load_variant(folder) for folder in args.variants})
    for name, snapshots in variants.items():
        vectors = piece_vectors(snapshots, assigned)
        pairs = lambda items: [float(vectors[a] @ vectors[b]) for a, b in items if a in vectors and b in vectors]
        negative = pairs(other_camera)
        crowd_same, crowd_other = crowd_similarity(snapshots, assigned)
        for x in tracklets:
            x.embedding = vectors.get(x.id)
        result = tr.link_tracklets(tracklets, tr.TrinityConfig(), zones)
        owner = {i: t.track_id for t in result.tracks for i in t.tracklets}
        numbers = score(assigned, assigned.tracklet.map(owner))
        errors = check_pairs(assigned, assigned.tracklet.map(owner))
        verified = len(SAME_PERSON) + len(DIFFERENT_PEOPLE) - sum(map(len, errors.values()))
        print(f"{name:38s} строгие {stats(pairs(sure), negative)}  связи {stats(pairs(sustained), negative)}  после паузы {stats(pairs(later), negative)}  "
              f"толпа {stats(crowd_same, crowd_other)}  склейка: {len(result.tracks)}, {numbers['соответствие']:.3f}, {verified}/{len(SAME_PERSON) + len(DIFFERENT_PEOPLE)}", flush=True)


if __name__ == "__main__":
    main()
