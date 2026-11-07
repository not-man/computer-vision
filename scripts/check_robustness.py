"""Устойчивость результата: пороги по одному на ±20%, отключение правил, сдвиг времени, порог детекции.

Считается в памяти: настройки и треки не меняются. Совпадение показывает стабильность, а не правильность:
результат может быть стабильно ошибочным, для этого нужен эталон.
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # варианты считаются параллельно, потоки внутри только мешают

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import linear_sum_assignment

from mct import tracklets as tracklets_module
from mct import trinity as T
from mct.geometry import load_cameras
from mct.plan import Plan
from mct.tracklets import build_tracklets, filter_config, load_perception

OFF = 1e9
SCALED_TRINITY = (
    "k", "min_inlier_share", "conflict_k", "conflict_min_overlap_s", "same_camera_overlap_s", "sustained_s", "brief_gap_s",
    "extrapolation_s", "extrapolation_sigma", "calibration_sigma", "v_max", "a_max", "direction_min_speed", "direction_min_cos",
    "score_margin", "gap_tolerance_cap_m", "reach_speed", "brief_max_m", "adjacent_gap_s", "adjacent_distance_m",
    "yellow_d_margin", "yellow_r_margin", "r_gap", "gap_veto", "r_hint", "hint_margin", "min_track_s",
    "weight_g", "weight_t", "weight_r", "contact_r", "contact_gap_s", "edge_override_r", "share_override_r", "share_override_min", "door_gap_veto", "return_window_s", "r_return", "return_margin",
)
SCALED_ZONES = (
    "door_buffer_s", "stationary_buffer_s", "occluder_buffer_s", "edge_out_buffer_s", "open_buffer_s",
    "stationary_radius", "edge_out_radius", "stationary_speed",
)
SCALED_FILTER = (
    "out_of_floor_m", "floor_sigmas", "low_height_m", "low_margin_m", "static_spread_m", "static_min_s",
    "edge_share_for_cut_person", "min_peak_conf", "static_box_share", "repeat_min_s", "repeat_max_s", "repeat_max_conf",
    "repeat_iou", "repeat_gap_s", "weak_median_conf", "weak_peak_conf", "weak_box_share", "weak_iou", "mirror_inside_share",
    "jump_max_cross", "jump_min_match", "jump_min_lead", "jump_near_m", "jump_near_s", "jump_apart_m", "swap_min_gain", "swap_min_match", "swap_touch_s",
)
CAMERAS = ("camera_new_1", "camera_new_2", "camera_new_3", "camera_new_4", "camera_new_5")
R_VETO_CURVE = (-0.2, -0.1, -0.05, 0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.3)
MIN_TRACK_CURVE = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 8.0, 10.0)


@dataclass(frozen=True)
class Case:
    name: str
    group: str
    filter: dict = field(default_factory=dict)
    trinity: dict = field(default_factory=dict)
    zones: dict = field(default_factory=dict)
    shift: tuple[str, float] | None = None
    min_conf: float | None = None
    door_radius: float | None = None
    no_doors: bool = False
    no_column: bool = False
    plain_embedding: bool = False


def build_cases(cfg: T.TrinityConfig, filters: tracklets_module.FilterConfig) -> list[Case]:
    cases = [Case("как сейчас", "база")]
    defaults = T.ZoneMap()
    for factor in (0.8, 1.2):
        cases += [Case(f"{n} ×{factor}", "пороги", trinity={n: getattr(cfg, n) * factor}) for n in SCALED_TRINITY]
        cases += [Case(f"{n} ×{factor}", "пороги", zones={n: getattr(defaults, n) * factor}) for n in SCALED_ZONES]
        cases += [Case(f"{n} ×{factor}", "пороги", filter={n: getattr(filters, n) * factor}) for n in SCALED_FILTER]
        cases.append(Case(f"радиус двери ×{factor}", "пороги", door_radius=factor))
    # R около нуля: ±20% ничего не значат, поэтому сдвиг абсолютный
    cases += [Case(f"r_veto={v}", "сдвиг R", trinity={"r_veto": v}) for v in (-0.05, 0.15)]
    cases += [Case(f"r_gap={v}", "сдвиг R", trinity={"r_gap": v}) for v in (0.1, 0.3)]
    cases += [Case(f"gap_veto={v}", "сдвиг R", trinity={"gap_veto": v}) for v in (0.05, 0.25)]
    cases += [Case(f"r_hint={v}", "сдвиг R", trinity={"r_hint": v}) for v in (0.5, 0.7)]
    objects_off = {"repeat_count": 10**6, "weak_count": 10**6}
    all_filters_off = {**objects_off, "mirrors": (), "out_of_floor_m": -OFF, "low_margin_m": -OFF, "static_min_s": OFF, "min_peak_conf": 0.0}
    cases += [
        Case("размеченное зеркало", "откл", filter={"mirrors": ()}),
        Case("предметы", "откл", filter=objects_off),
        Case("вне пола", "откл", filter={"out_of_floor_m": -OFF}),
        Case("у границы и низкий", "откл", filter={"low_margin_m": -OFF}),
        Case("мебель", "откл", filter={"static_min_s": OFF}),
        Case("слабая уверенность", "откл", filter={"min_peak_conf": 0.0}),
        Case("все фильтры ложных", "откл", filter=all_filters_off),
        Case("почти стык", "откл", trinity={"adjacent_gap_s": -1.0}),
        Case("разрез при перескоке трекера", "откл", filter={"jump_max_cross": -OFF}),
        Case("разрез при обмене номерами", "откл", filter={"swap_min_gain": OFF}),
        Case("один объект на камере", "откл", trinity={"contact_r": OFF}),
        Case("короткий контакт без края", "откл", trinity={"edge_override_r": OFF}),
        Case("доля точек при сходстве", "откл", trinity={"share_override_r": OFF}),
        Case("строже у двери", "откл", trinity={"door_gap_veto": 0.15}),
        Case("возврат до минуты", "откл", trinity={"r_return": OFF}),
        Case("предел допуска на разрыве", "откл", trinity={"gap_tolerance_cap_m": OFF}),
        Case("предел при контакте", "откл", trinity={"brief_max_m": OFF}),
        Case("смена номера на камере", "откл", trinity={"same_camera_overlap_s": -1.0}),
        Case("зоны дверей", "откл", no_doors=True),
        Case("колонна", "откл", no_column=True),
        Case("вычитание среднего камеры", "откл", plain_embedding=True),
    ]
    cases += [Case(f"время cam_{c[-1]} {dt:+.1f} с", "время", shift=(c, dt)) for c in CAMERAS for dt in (-0.2, 0.2)]
    cases += [Case(f"порог детекции {c}", "детектор", min_conf=c) for c in (0.15, 0.25, 0.35)]
    cases += [Case(f"r_veto {v}", "кривая r_veto", trinity={"r_veto": v}) for v in R_VETO_CURVE]
    cases += [Case(f"длительность {v}", "кривая длительности", trinity={"min_track_s": v}) for v in MIN_TRACK_CURVE]
    return cases


class Context:
    observations: pd.DataFrame
    embeddings: dict
    plan: Plan
    extra_zones: list[T.Zone]
    filters: tracklets_module.FilterConfig
    trinity = T.TrinityConfig()


def plain_mean_embeddings(items: list[T.Tracklet], appearance: dict[int, np.ndarray]) -> None:
    for item in items:
        if item.id in appearance:
            mean = appearance[item.id].mean(axis=0)
            item.embedding = mean / max(float(np.linalg.norm(mean)), 1e-12)


def run_case(case: Case) -> dict:
    observations = Context.observations
    if case.shift:
        camera, delta = case.shift
        observations = observations.copy()
        chosen = observations.camera == camera
        observations.loc[chosen, "t_s"] = observations.loc[chosen, "t_s"] + delta
    if case.min_conf:
        observations = observations[observations.conf >= case.min_conf]

    original = tracklets_module._set_embeddings
    if case.plain_embedding:
        tracklets_module._set_embeddings = plain_mean_embeddings
    try:
        tracklets, assigned, dropped = build_tracklets(observations, Context.embeddings, replace(Context.filters, **case.filter))
    finally:
        tracklets_module._set_embeddings = original

    doors = [] if case.no_doors else Context.plan.door_zones()
    if case.door_radius:
        doors = [replace(zone, r=zone.r * case.door_radius) for zone in doors]
    zones = T.ZoneMap(doors + ([] if case.no_column else Context.extra_zones), **case.zones)
    result = T.link_tracklets(tracklets, replace(Context.trinity, **case.trinity), zones)

    track_of = {i: t.track_id for t in result.tracks for i in t.tracklets}
    table = assigned[assigned.source != "unreliable"][["camera", "frame", "raw_track", "tracklet"]].drop_duplicates(["camera", "frame", "raw_track"]).copy()
    table["track"] = table.tracklet.map(track_of)
    links = [link for track in result.tracks for link in track.links]
    return {
        "name": case.name, "group": case.group, "tracks": len(result.tracks),
        "hints": sum(len(t.hints) for t in result.tracks), "yellow": sum(link.confidence == "yellow" for link in links),
        "sustained": sum(link.tier == "sustained" for link in links), "brief": sum(link.tier == "brief" for link in links),
        "gap": sum(link.tier == "gap" for link in links), "dropped": len(dropped), "table": table.dropna(subset=["track"]),
    }


def agreement(base: pd.DataFrame, other: pd.DataFrame) -> float:
    """Как IDF1 относительно текущего результата: доля наблюдений, оставшихся у того же человека."""
    merged = base.merge(other, on=["camera", "frame", "raw_track"], suffixes=("_a", "_b"))
    if merged.empty:
        return 0.0
    counts = merged.groupby(["track_a", "track_b"]).size().reset_index(name="n")
    rows = {value: i for i, value in enumerate(counts.track_a.unique())}
    columns = {value: i for i, value in enumerate(counts.track_b.unique())}
    overlap = np.zeros((len(rows), len(columns)))
    overlap[counts.track_a.map(rows), counts.track_b.map(columns)] = counts.n
    row, column = linear_sum_assignment(-overlap)
    return float(2 * overlap[row, column].sum() / (len(base) + len(other)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("perception", type=Path)
    parser.add_argument("--out", type=Path, default=Path("artifacts/tracks/robustness.csv"))
    parser.add_argument("--workers", type=int, default=14)
    args = parser.parse_args()

    Context.plan = Plan.load()
    Context.observations, Context.embeddings = load_perception(args.perception, load_cameras(), Context.plan)
    Context.extra_zones = [T.Zone(**zone) for zone in yaml.safe_load(Path("configs/zones.yaml").read_text())["zones"]]
    Context.filters = filter_config()

    cases = build_cases(Context.trinity, Context.filters)
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(run_case, cases))
    base = results[0]["table"]
    rows = [{**{k: v for k, v in r.items() if k != "table"}, "agreement": round(agreement(base, r["table"]), 3)} for r in results]
    table = pd.DataFrame(rows)
    table["delta"] = table.tracks - rows[0]["tracks"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)

    scaled = table[table.group == "пороги"]
    print(json.dumps({
        "вариантов": len(table), "треков_сейчас": rows[0]["tracks"],
        "±20%": {"вариантов": len(scaled), "треков": [int(scaled.tracks.min()), int(scaled.tracks.max())], "совпадение_min": float(scaled.agreement.min()),
                 "без_изменения_числа": int((scaled.delta == 0).sum())},
    }, ensure_ascii=False))
    columns = ["name", "tracks", "delta", "agreement", "yellow"]
    for group in ("откл", "сдвиг R", "время", "детектор", "кривая r_veto", "кривая длительности"):
        print(f"--- {group}")
        print(table[table.group == group][columns].to_string(index=False))


if __name__ == "__main__":
    main()
