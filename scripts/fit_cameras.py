"""Подбор физических моделей камер по стенам плана, столу известного размера и вертикальным краям."""
import argparse
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import least_squares

from mct.geometry import CONFIG, Camera, Lens, edge_chains, load_cameras, straightness

SIGMA_POINT_PX = 6.0
SIGMA_LINE_PX = 5.0
SIGMA_VERTICAL_PX = 3.0
SIGMA_TABLE_PX = 3.0
SIGMA_POSITION_M = 0.8
PRIOR_HEIGHT_M, SIGMA_HEIGHT_M = 3.0, 0.8

# Люди: один человек на двух камерах стоит в одной точке плана и имеет один рост
PEOPLE_STEP_S = 0.5
PEOPLE_MIN_OVERLAP_S = 1.0
PEOPLE_MAX_TIME_ERROR_S = 0.06
PEOPLE_FULL_WEIGHT_SAMPLES = 1000  # больше примеров не должны перевешивать стены плана
PEOPLE_SIGMA_BASE_M, PEOPLE_SIGMA_PER_M = 0.25, 0.03
SIGMA_SAME_HEIGHT_M = 0.12
NEAR_FOR_HEIGHT_M = 12.0
TYPICAL_PERSON_M, SIGMA_TYPICAL_PERSON_M = 1.72, 0.03


def find_vertical_edges(lens: Lens, image: np.ndarray, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Прямые края, сходящиеся в одну точку ниже центра кадра: это вертикали мира, а не швы плитки."""
    segments = []
    for chain in edge_chains(image, min_extent=60):
        points = lens.undistort(chain)
        ratio, length = straightness(points)
        if ratio > 0.005 or length < 50:
            continue
        center = points.mean(axis=0)
        _, vectors = np.linalg.eigh(np.cov((points - center).T))
        direction = vectors[:, 1]
        segments.append((center - direction * length / 2, center + direction * length / 2, length))
    if len(segments) < 2:
        return []

    starts = np.array([s[0] for s in segments])
    ends = np.array([s[1] for s in segments])
    lengths = np.array([s[2] for s in segments])
    centers = (starts + ends) / 2
    lines = np.cross(np.column_stack([starts, np.ones(len(starts))]), np.column_stack([ends, np.ones(len(ends))]))
    rng = np.random.default_rng(seed)
    best_score, best_inliers = 0.0, np.zeros(len(segments), bool)
    for _ in range(4000):
        i, j = rng.choice(len(segments), 2, replace=False)
        vp = np.cross(lines[i], lines[j])
        if abs(vp[2]) < 1e-9:
            continue
        vp = vp[:2] / vp[2]
        if not (lens.cy + 200 < vp[1] < lens.cy + 5000 and abs(vp[0] - lens.cx) < 2500):
            continue
        # Отклонение концов отрезка от прямой, проведённой из точки схода через его середину
        to_center = centers - vp
        normal = np.column_stack([-to_center[:, 1], to_center[:, 0]])
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        deviation = np.abs(((starts - centers) * normal).sum(axis=1))
        inliers = deviation < 1.5
        score = float(lengths[inliers].sum())
        if score > best_score:
            best_score, best_inliers = score, inliers
    return [segments[i] for i in np.nonzero(best_inliers)[0]]


@dataclass
class PeopleSamples:
    """Одновременные наблюдения одного человека двумя камерами, в кадре без дисторсии."""

    camera_a: str
    camera_b: str
    foot_a: np.ndarray
    head_a: np.ndarray
    foot_b: np.ndarray
    head_b: np.ndarray
    track: np.ndarray


def nearest_index(sorted_times: np.ndarray, times: np.ndarray) -> np.ndarray:
    right = np.clip(np.searchsorted(sorted_times, times), 0, len(sorted_times) - 1)
    left = np.maximum(right - 1, 0)
    return np.where(np.abs(sorted_times[left] - times) <= np.abs(sorted_times[right] - times), left, right)


def collect_people(tracks_dir: Path) -> list[PeopleSamples]:
    """Пары наблюдений одного человека. Треклеты со спорными связями не участвуют: там могли поменяться люди."""
    observations = pd.read_parquet(tracks_dir / "observations.parquet")
    usable = observations[(observations.source == "foot") & ~observations.touches_edge]
    paths = {tracklet: group.sort_values("t_s") for tracklet, group in usable.groupby("tracklet")}
    rows: dict[tuple[str, str], list[np.ndarray]] = {}
    for track in json.loads((tracks_dir / "tracks.json").read_text(encoding="utf-8"))["tracks"]:
        doubtful = {end for link in track["links"] if link["confidence"] == "yellow" for end in (link["a"], link["b"])}
        present = [tracklet for tracklet in track["tracklets"] if tracklet in paths and tracklet not in doubtful]
        for first, second in combinations(present, 2):
            a, b = sorted((paths[first], paths[second]), key=lambda path: path.camera.iloc[0])
            if a.camera.iloc[0] == b.camera.iloc[0]:
                continue
            start, end = max(a.t_s.iloc[0], b.t_s.iloc[0]), min(a.t_s.iloc[-1], b.t_s.iloc[-1])
            if end - start < PEOPLE_MIN_OVERLAP_S:
                continue
            times = np.arange(start, end, PEOPLE_STEP_S)
            ta, tb = a.t_s.to_numpy(), b.t_s.to_numpy()
            ia, ib = nearest_index(ta, times), nearest_index(tb, times)
            good = (np.abs(ta[ia] - times) <= PEOPLE_MAX_TIME_ERROR_S) & (np.abs(tb[ib] - times) <= PEOPLE_MAX_TIME_ERROR_S)
            if good.any():
                columns = ["foot_u", "foot_v", "head_u", "head_v"]
                rows.setdefault((a.camera.iloc[0], b.camera.iloc[0]), []).append(np.column_stack([
                    a[columns].to_numpy()[ia[good]], b[columns].to_numpy()[ib[good]], np.full(good.sum(), track["track_id"]),
                ]))
    samples = []
    for (camera_a, camera_b), parts in sorted(rows.items()):
        table = np.vstack(parts)
        samples.append(PeopleSamples(camera_a, camera_b, table[:, :2], table[:, 2:4], table[:, 4:6], table[:, 6:8], table[:, 8].astype(int)))
    return samples


def select_tracks(samples: list[PeopleSamples], remainder: int) -> list[PeopleSamples]:
    """Половина треков по чётности номера: чтобы проверять подбор на примерах, которых он не видел."""
    chosen = []
    for item in samples:
        keep = item.track % 2 == remainder
        if keep.any():
            chosen.append(PeopleSamples(item.camera_a, item.camera_b, item.foot_a[keep], item.head_a[keep], item.foot_b[keep], item.head_b[keep], item.track[keep]))
    return chosen


def people_report(samples: list[PeopleSamples], cameras: dict[str, Camera]) -> dict:
    """Расхождение положений и рост: медианы по парам и камерам."""
    distances, heights = {}, {}
    for item in samples:
        a, b = cameras[item.camera_a], cameras[item.camera_b]
        gap = a.rectified_to_floor(item.foot_a) - b.rectified_to_floor(item.foot_b)
        distances[f"{item.camera_a[-1]}{item.camera_b[-1]}"] = np.hypot(gap[:, 0], gap[:, 1])
        heights.setdefault(item.camera_a[-1], []).append(a.heights_of(item.foot_a, item.head_a))
        heights.setdefault(item.camera_b[-1], []).append(b.heights_of(item.foot_b, item.head_b))
    return {
        "расстояние_м": {pair: round(float(np.median(values)), 2) for pair, values in distances.items()},
        "все_м": round(float(np.median(np.concatenate(list(distances.values())))), 2),
        "рост_м": {camera: round(float(np.median(np.concatenate(values))), 2) for camera, values in sorted(heights.items())},
    }


def build_camera(lens: Lens, f: float, values: np.ndarray) -> Camera:
    return Camera(lens, f, *[float(v) for v in values])


def line_distance(camera: Camera, plan_line_m: np.ndarray, points: np.ndarray) -> np.ndarray:
    a, b = camera.project_h(np.column_stack([plan_line_m, np.zeros(2)]))
    line = np.cross(a, b)
    return (points @ line[:2] + line[2]) / np.linalg.norm(line[:2])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("configs/plan_features.yaml"))
    parser.add_argument("--lenses", type=Path, default=Path("artifacts/geometry/lenses.json"))
    parser.add_argument("--backgrounds", type=Path, default=Path("artifacts/geometry/background"))
    parser.add_argument("--out", type=Path, default=CONFIG)
    parser.add_argument("--people", type=Path, default=None, help="папка треков: добавить совпадения людей между камерами")
    parser.add_argument("--holdout", action="store_true", help="подбирать на половине треков и проверять на другой половине")
    args = parser.parse_args()

    features = yaml.safe_load(args.features.read_text(encoding="utf-8"))
    scale = features["plan_px_per_m"]
    lenses = {name: Lens(**values) for name, values in json.loads(args.lenses.read_text(encoding="utf-8")).items()}
    names = list(features["cameras"])
    # У камер одной модели фокус общий
    focus_groups = [["camera_new_1", "camera_new_2", "camera_new_3", "camera_new_5"], ["camera_new_4"]]
    focus_index = {name: g for g, group in enumerate(focus_groups) for name in group}

    data = {}
    for name in names:
        lens, spec = lenses[name], features["cameras"][name]
        verticals = find_vertical_edges(lens, cv2.imread(str(args.backgrounds / f"{name}.png")))
        data[name] = {
            "lens": lens,
            "position": np.array(spec["position_plan"]) / scale,
            "points_image": lens.undistort(np.array([p["image"] for p in spec.get("points", [])], float)) if spec.get("points") else np.empty((0, 2)),
            "points_plan": np.array([p["plan"] for p in spec.get("points", [])], float).reshape(-1, 2) / scale,
            "lines": [(lens.undistort(np.array(l["image"], float)), np.array(l["plan"], float) / scale) for l in spec.get("lines", [])],
            "verticals": verticals,
        }
        print(f"{name}: вертикальных краёв {len(verticals)}", flush=True)

    table = features["table"]
    table_corners = {name: lenses[name].undistort(np.array(c, float)) for name, c in table["corners"].items()}
    half_x, half_y = table["size_m"][0] / 2, table["size_m"][1] / 2
    local = np.array([[half_x, -half_y], [-half_x, -half_y], [-half_x, half_y], [half_x, half_y]])  # СВ, СЗ, ЮЗ, ЮВ

    x0 = []
    for name in names:
        spec = features["cameras"][name]
        x0 += [*data[name]["position"], PRIOR_HEIGHT_M, np.radians(spec["yaw_deg"]), np.radians(40.0), 0.0]
    x0 += [650.0, 650.0]  # фокусы групп
    x0 += [*(np.array([445.0, 335.0]) / scale), 0.0]  # стол: центр и поворот
    x0 = np.array(x0)
    n = len(names)

    samples = collect_people(args.people) if args.people else []
    held_out = select_tracks(samples, 1) if args.holdout else []
    if args.holdout:
        samples = select_tracks(samples, 0)
    total_samples = sum(len(item.track) for item in samples)
    people_scale = min(1.0, float(np.sqrt(PEOPLE_FULL_WEIGHT_SAMPLES / max(total_samples, 1))))
    # Вес и близость считаются один раз по текущим камерам: число невязок не должно меняться при подборе
    current = load_cameras()
    prepared = []
    for item in samples:
        a, b = current[item.camera_a], current[item.camera_b]
        floor_a, floor_b = a.rectified_to_floor(item.foot_a), b.rectified_to_floor(item.foot_b)
        distance_a = np.hypot(floor_a[:, 0] - a.x, floor_a[:, 1] - a.y)
        distance_b = np.hypot(floor_b[:, 0] - b.x, floor_b[:, 1] - b.y)
        # Вдали положение неточно, поэтому вес падает с расстоянием до камер
        prepared.append((item, PEOPLE_SIGMA_BASE_M + PEOPLE_SIGMA_PER_M * (distance_a + distance_b), (distance_a < NEAR_FOR_HEIGHT_M) & (distance_b < NEAR_FOR_HEIGHT_M)))
    if samples:
        print(f"люди: {total_samples} примеров в подборе, вес {people_scale:.2f}", flush=True)

    def camera_of(x: np.ndarray, index: int) -> Camera:
        name = names[index]
        return build_camera(data[name]["lens"], float(x[6 * n + focus_index[name]]), x[6 * index:6 * index + 6])

    def residuals(x: np.ndarray) -> np.ndarray:
        out = []
        for index, name in enumerate(names):
            camera, d = camera_of(x, index), data[name]
            if len(d["points_image"]):
                projected = camera.project(np.column_stack([d["points_plan"], np.zeros(len(d["points_plan"]))]))
                out.append(((projected - d["points_image"]) / SIGMA_POINT_PX).ravel())
            for image_points, plan_line in d["lines"]:
                out.append(line_distance(camera, plan_line, image_points) / SIGMA_LINE_PX)
            nadir = camera.nadir()
            for start, end, length in d["verticals"]:
                center = (start + end) / 2
                direction = center - nadir
                normal = np.array([-direction[1], direction[0]]) / np.linalg.norm(direction)
                out.append(np.array([(start - center) @ normal]) * np.sqrt(length / 100) / SIGMA_VERTICAL_PX)
            out.append((x[6 * index:6 * index + 2] - d["position"]) / SIGMA_POSITION_M)
            out.append(np.array([(x[6 * index + 2] - PRIOR_HEIGHT_M) / SIGMA_HEIGHT_M]))

        near_heights: dict[str, list[np.ndarray]] = {}
        for item, sigma, near in prepared:
            a, b = camera_of(x, names.index(item.camera_a)), camera_of(x, names.index(item.camera_b))
            floor_a, floor_b = a.rectified_to_floor(item.foot_a), b.rectified_to_floor(item.foot_b)
            out.append(people_scale * ((floor_a - floor_b) / sigma[:, None]).ravel())
            if near.any():
                height_a = a.heights_of(item.foot_a[near], item.head_a[near])
                height_b = b.heights_of(item.foot_b[near], item.head_b[near])
                out.append(people_scale * (height_a - height_b) / SIGMA_SAME_HEIGHT_M)
                near_heights.setdefault(item.camera_a, []).append(height_a)
                near_heights.setdefault(item.camera_b, []).append(height_b)
        # Средний рост задаёт масштаб: без него высоту камеры подбор менял бы почти свободно
        for parts in near_heights.values():
            out.append(np.array([(np.median(np.concatenate(parts)) - TYPICAL_PERSON_M) / SIGMA_TYPICAL_PERSON_M]))

        center, angle = x[6 * n + 2:6 * n + 4], x[6 * n + 4]
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        table_world = np.column_stack([center + local @ rotation.T, np.full(4, table["height_m"])])
        for name, corners in table_corners.items():
            camera = camera_of(x, names.index(name))
            out.append(((camera.project(table_world) - corners) / SIGMA_TABLE_PX).ravel())
        return np.concatenate(out)

    result = least_squares(residuals, x0, loss="soft_l1", f_scale=1.0, max_nfev=400)
    x = result.x
    print("сходимость:", result.status, "итераций:", result.nfev, "стоимость:", round(float(result.cost), 1), flush=True)
    if samples:
        after = {name: camera_of(x, index) for index, name in enumerate(names)}
        print("в подборе, было:", json.dumps(people_report(samples, current), ensure_ascii=False), flush=True)
        print("в подборе, стало:", json.dumps(people_report(samples, after), ensure_ascii=False), flush=True)
        if held_out:
            print("отложено, было:", json.dumps(people_report(held_out, current), ensure_ascii=False), flush=True)
            print("отложено, стало:", json.dumps(people_report(held_out, after), ensure_ascii=False), flush=True)

    config = {"plan_px_per_m": scale, "cameras": {}}
    for index, name in enumerate(names):
        camera, d = camera_of(x, index), data[name]
        report = {
            "plan_px": [round(camera.x * scale), round(camera.y * scale)], "высота_м": round(camera.height, 2),
            "yaw": round(float(np.degrees(camera.yaw)), 1), "tilt": round(float(np.degrees(camera.tilt)), 1),
            "roll": round(float(np.degrees(camera.roll)), 1), "f": round(camera.f, 1),
        }
        if len(d["points_image"]):
            projected = camera.project(np.column_stack([d["points_plan"], np.zeros(len(d["points_plan"]))]))
            report["точки_px"] = round(float(np.sqrt(((projected - d["points_image"]) ** 2).sum(axis=1).mean())), 1)
        if d["lines"]:
            distances = np.concatenate([line_distance(camera, plan, image) for image, plan in d["lines"]])
            report["линии_px"] = round(float(np.sqrt((distances**2).mean())), 1)
        print(name, json.dumps(report, ensure_ascii=False), flush=True)
        config["cameras"][name] = {
            "lens": {"k1": d["lens"].k1, "k2": d["lens"].k2, "cx": d["lens"].cx, "cy": d["lens"].cy},
            "f": round(camera.f, 2), "x": round(camera.x, 4), "y": round(camera.y, 4),
            "height": round(camera.height, 4), "yaw": round(camera.yaw, 5),
            "tilt": round(camera.tilt, 5), "roll": round(camera.roll, 5),
        }

    center, angle = x[6 * n + 2:6 * n + 4], x[6 * n + 4]
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    table_world = np.column_stack([center + local @ rotation.T, np.full(4, table["height_m"])])
    for name, corners in table_corners.items():
        camera = camera_of(x, names.index(name))
        error = np.sqrt(((camera.project(table_world) - corners) ** 2).sum(axis=1).mean())
        print(f"стол на {name}: {error:.1f} px", flush=True)
    print("стол на плане, px:", np.round(center * scale).astype(int).tolist(), "поворот, град:", round(float(np.degrees(angle)), 1), flush=True)
    config["table"] = {"x": round(float(center[0]), 3), "y": round(float(center[1]), 3), "angle": round(float(angle), 4)}

    args.out.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()
