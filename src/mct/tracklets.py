"""Из наблюдений детектора в треклеты на плане: положение, погрешность, резка и фильтр ложных детекций."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pycocotools import mask as mask_utils

from mct.geometry import CONFIG as CAMERAS_CONFIG
from mct.geometry import IMAGE_SIZE, Camera
from mct.plan import FEATURES, Plan
from mct.trinity import Tracklet

LOCATE_VERSION = 4  # увеличить при изменении locate: кэш положений пересчитается
MIN_PIXELS = 50
SUBSAMPLE = 3  # каждый третий пиксель маски: точность та же, считается быстрее
FRAME_MARGIN_PX = 6  # ближе к краю кадра точка считается обрезанной
MIN_FOOT_HEIGHT_M = 1.35  # ниже этого ноги, скорее всего, закрыты: положение берём по макушке
ASSUMED_HEIGHT_M = 1.72
HEIGHT_SIGMA_M = 0.12
UNRELIABLE_SIGMA_M = 2.0
MAX_HEAD_DISTANCE_M = 30.0

JUMP_SPEED = 5.0  # м/с: быстрее не ходят, значит, трекер перепутал людей
MAX_TIME_GAP_S = 4.0
UNRELIABLE_NEIGHBOUR_S = 0.5
SMOOTH_WINDOW = 5
CUT_WINDOW = 9


def inside_share(box: np.ndarray, region: list[float]) -> float:
    """Какая доля рамки лежит внутри области."""
    width = max(0.0, min(box[2], region[2]) - max(box[0], region[0]))
    height = max(0.0, min(box[3], region[3]) - max(box[1], region[1]))
    return width * height / max((box[2] - box[0]) * (box[3] - box[1]), 1e-9)


@dataclass(frozen=True)
class FilterConfig:
    out_of_floor_m: float = -0.5
    floor_sigmas: float = 1.5  # вдали положение неточно, поэтому допуск по полу растёт вместе с погрешностью
    low_height_m: float = 1.3
    low_margin_m: float = -0.15  # у людей вблизи с закрытыми ногами точка заходит за границу на сантиметры, у зеркала дальше
    static_spread_m: float = 0.3
    static_min_s: float = 3.0
    edge_share_for_cut_person: float = 0.5
    min_peak_conf: float = 0.35
    # Предмет, похожий на человека: детектор возвращается к нему в той же рамке снова и снова
    static_box_share: float = 0.15  # разброс центра рамки относительно её высоты
    repeat_min_s: float = 0.5  # по совсем коротким обломкам о неподвижности судить нельзя
    repeat_max_s: float = 20.0
    repeat_max_conf: float = 0.7  # долго и уверенно на одном месте стоит настоящий человек
    repeat_iou: float = 0.5
    repeat_count: int = 2  # сколько ещё раз, кроме этого
    repeat_gap_s: float = 20.0
    # Слабому куску достаточно одного повторения: уверенные куски в той же рамке чаще всего настоящие люди вдали
    weak_median_conf: float = 0.45
    weak_peak_conf: float = 0.7
    weak_box_share: float = 0.25
    weak_iou: float = 0.4
    weak_count: int = 1
    # Известные зеркала: области на кадре. Отражение лежит внутри целиком, а человек рядом выходит за край
    mirrors: tuple[tuple[str, tuple[float, float, float, float]], ...] = ()
    mirror_inside_share: float = 0.8


def _near_frame_edge(point: np.ndarray) -> bool:
    return bool(
        point[0] < FRAME_MARGIN_PX or point[1] < FRAME_MARGIN_PX
        or point[0] > IMAGE_SIZE[0] - FRAME_MARGIN_PX or point[1] > IMAGE_SIZE[1] - FRAME_MARGIN_PX
    )


def _spread_covariance(floor_of, image_point: np.ndarray, along: np.ndarray, sigma_along: float, sigma_across: float) -> np.ndarray:
    """Погрешность на плане от погрешности в пикселях исходного кадра: поэтому дальние люди получают большую."""
    base = floor_of(image_point)
    across = np.array([-along[1], along[0]])
    a = floor_of(image_point + along * sigma_along) - base
    b = floor_of(image_point + across * sigma_across) - base
    return np.outer(a, a) + np.outer(b, b)


def locate(row_rle: str, camera: Camera) -> dict | None:
    """Положение одного наблюдения на полу: по ногам или, если они закрыты, по макушке."""
    mask = mask_utils.decode({"size": [IMAGE_SIZE[1], IMAGE_SIZE[0]], "counts": row_rle.encode()})
    ys, xs = np.nonzero(mask)
    if len(xs) < MIN_PIXELS:
        return None
    foot, head = camera.foot_and_head(np.column_stack([xs[::SUBSAMPLE], ys[::SUBSAMPLE]]))
    lens = camera.lens
    foot_image, head_image = lens.distort(foot)[0], lens.distort(head)[0]
    height = camera.height_of(foot, head)
    span_px = float(np.linalg.norm(head_image - foot_image))
    sigma_along = 4.0 + 0.03 * span_px
    sigma_across = 3.0
    along = (head_image - foot_image) / max(span_px, 1e-6)

    # Отступ от границы пола считаем только по ногам: у отражения и экрана они за стеной,
    # а закрытые ноги человека смещают точку к камере, то есть внутрь пола
    foot_floor = camera.rectified_to_floor(foot)[0]
    foot_cut, head_cut = _near_frame_edge(foot_image), _near_frame_edge(head_image)
    source = "foot"
    if foot_cut and head_cut:
        source = "unreliable"
    elif (foot_cut or height < MIN_FOOT_HEIGHT_M) and not head_cut:
        source = "head"
    if source == "head":
        floor = camera.floor_from_head(head)
        if np.hypot(floor[0] - camera.x, floor[1] - camera.y) > MAX_HEAD_DISTANCE_M:
            source = "unreliable"

    if source == "foot":
        floor = foot_floor
        cov = _spread_covariance(lambda p: camera.image_to_floor(p)[0], foot_image, along, sigma_along, sigma_across)
    elif source == "head":
        cov = _spread_covariance(lambda p: camera.floor_from_head(lens.undistort(p)[0]), head_image, along, sigma_along, sigma_across)
        shift = camera.floor_from_head(head, ASSUMED_HEIGHT_M + HEIGHT_SIGMA_M) - floor
        cov = cov + np.outer(shift, shift)
    else:
        floor = foot_floor
        cov = np.eye(2) * UNRELIABLE_SIGMA_M**2
    return {
        "X": float(floor[0]), "Y": float(floor[1]), "cov_xx": float(cov[0, 0]), "cov_xy": float(cov[0, 1]),
        "cov_yy": float(cov[1, 1]), "source": source, "person_height": float(height),
        "foot_x": float(foot_floor[0]), "foot_y": float(foot_floor[1]),
        "foot_u": float(foot[0]), "foot_v": float(foot[1]), "head_u": float(head[0]), "head_v": float(head[1]),
    }


def locate_observations(frame: pd.DataFrame, camera: Camera, plan: Plan) -> pd.DataFrame:
    located = []
    for row in frame.itertuples():
        values = locate(row.mask_rle, camera)
        if values is None:
            continue
        values.update({
            "camera": row.camera, "frame": row.frame, "t_s": row.t_s, "raw_track": row.raw_track, "conf": row.conf,
            "touches_edge": row.touches_edge, "margin": plan.floor_margin(np.array([values["foot_x"], values["foot_y"]])),
            "x1": row.x1, "y1": row.y1, "x2": row.x2, "y2": row.y2,
        })
        located.append(values)
    return pd.DataFrame(located)


def _cut_points(piece: pd.DataFrame) -> np.ndarray:
    """Куда резать трек: устойчивый скачок быстрее человека или слишком долгая пауза.

    Скачок ищем по сглаженному пути: ноги часто дёргаются на кадр-другой из-за стола или соседа,
    а настоящая подмена человека сдвигает весь дальнейший путь.
    """
    path = piece[["X", "Y"]].rolling(CUT_WINDOW, center=True, min_periods=1).median().to_numpy()
    dt = np.diff(piece.t_s.to_numpy())
    step = np.hypot(*np.diff(path, axis=0).T)
    variance = piece.cov_xx.to_numpy() + piece.cov_yy.to_numpy()
    allowance = JUMP_SPEED * dt + 3 * np.sqrt(variance[:-1] + variance[1:])
    return np.nonzero((step > allowance) | (dt > MAX_TIME_GAP_S))[0] + 1


def _rejection(piece: pd.DataFrame, cfg: FilterConfig) -> str | None:
    box = piece[["x1", "y1", "x2", "y2"]].median().to_numpy()
    if any(camera == piece.camera.iloc[0] and inside_share(box, region) >= cfg.mirror_inside_share for camera, region in cfg.mirrors):
        return "зеркало: рамка внутри размеченной области"
    margin = float(piece.margin.median())
    height = float(piece.person_height.median())
    sigma = float(np.sqrt(np.median(piece.cov_xx + piece.cov_yy)))
    if piece.conf.max() < cfg.min_peak_conf:
        return "слабая уверенность"
    if margin < cfg.out_of_floor_m - cfg.floor_sigmas * sigma:
        return "вне пола: отражение или экран"
    cut_person = float(piece.touches_edge.mean()) >= cfg.edge_share_for_cut_person
    if cut_person:
        return None  # человек у края кадра обрезан, рост по нему ни о чём не говорит
    if margin < cfg.low_margin_m and height < cfg.low_height_m:
        return "на границе пола и низкий: напольное зеркало"
    duration = float(piece.t_s.iloc[-1] - piece.t_s.iloc[0])
    spread = float(max(piece.X.quantile(0.9) - piece.X.quantile(0.1), piece.Y.quantile(0.9) - piece.Y.quantile(0.1)))
    if duration >= cfg.static_min_s and spread < cfg.static_spread_m and height < cfg.low_height_m:
        return "не движется и низкий: мебель"
    return None


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def _repeated_objects(pieces: list[tuple[str, pd.DataFrame]], cfg: FilterConfig) -> set[int]:
    """Номера кусков, где детектор снова и снова видит «человека» неподвижно в той же рамке: это предмет."""
    still = []
    for index, (camera, piece) in enumerate(pieces):
        duration = float(piece.t_s.iloc[-1] - piece.t_s.iloc[0])
        weak = piece.conf.median() <= cfg.weak_median_conf and piece.conf.max() <= cfg.weak_peak_conf
        if not cfg.repeat_min_s <= duration <= cfg.repeat_max_s or piece.conf.median() > cfg.repeat_max_conf:
            continue
        box = piece[["x1", "y1", "x2", "y2"]].median().to_numpy()
        center_x, center_y = (piece.x1 + piece.x2) / 2, (piece.y1 + piece.y2) / 2
        spread = max(center_x.quantile(0.9) - center_x.quantile(0.1), center_y.quantile(0.9) - center_y.quantile(0.1))
        if spread <= (cfg.weak_box_share if weak else cfg.static_box_share) * (box[3] - box[1]):
            still.append((index, camera, box, float(piece.t_s.iloc[0]), weak))
    repeated = set()
    for index, camera, box, start, weak in still:
        matches = sum(
            1 for other, other_camera, other_box, other_start, _ in still
            if other != index and other_camera == camera and abs(other_start - start) >= cfg.repeat_gap_s
            and _box_iou(box, other_box) >= (cfg.weak_iou if weak else cfg.repeat_iou)
        )
        if matches >= (cfg.weak_count if weak else cfg.repeat_count):
            repeated.add(index)
    return repeated


def build_tracklets(
    observations: pd.DataFrame,
    embeddings: dict[str, tuple[np.ndarray, np.ndarray]],
    cfg: FilterConfig = FilterConfig(),
) -> tuple[list[Tracklet], pd.DataFrame, pd.DataFrame]:
    """Возвращает треклеты, наблюдения с номером треклета и таблицу отброшенного с причинами."""
    tracklets, assigned, dropped = [], [], []
    appearance: dict[int, np.ndarray] = {}
    next_id = 1
    observations = observations.sort_values(["camera", "raw_track", "t_s"])
    pieces = []
    for (camera, raw_track), everything in observations.groupby(["camera", "raw_track"], sort=True):
        everything = everything.reset_index(drop=True)
        # Положение без ног и макушки в кадре неизвестно: такие наблюдения не участвуют в склейке,
        # но подписываются тем же треком, если рядом по времени есть надёжные
        group = everything[everything.source != "unreliable"].reset_index(drop=True)
        if group.empty:
            dropped.append(_dropped_row(camera, raw_track, everything, "нет надёжного положения"))
            continue
        bounds = [0, *_cut_points(group).tolist(), len(group)]
        pieces += [(camera, raw_track, everything, group.iloc[start:end].copy()) for start, end in zip(bounds, bounds[1:])]

    repeated = _repeated_objects([(camera, piece) for camera, _, _, piece in pieces], cfg)
    for index, (camera, raw_track, everything, piece) in enumerate(pieces):
        reason = _rejection(piece, cfg) or ("предмет: снова и снова на одном месте" if index in repeated else None)
        if reason is not None:
            dropped.append(_dropped_row(camera, raw_track, piece, reason))
            continue
        xy = piece[["X", "Y"]].rolling(SMOOTH_WINDOW, center=True, min_periods=1).median().to_numpy()
        cov = np.stack([piece.cov_xx, piece.cov_xy, piece.cov_xy, piece.cov_yy], axis=1).reshape(-1, 2, 2)
        if camera in embeddings:
            keys, vectors = embeddings[camera]
            if len(vectors):
                chosen = (keys[:, 1] == raw_track) & np.isin(keys[:, 0], piece.frame.to_numpy())
                if chosen.any():
                    appearance[next_id] = vectors[chosen]
        first, last = piece.t_s.iloc[0], piece.t_s.iloc[-1]
        around = everything[(everything.t_s >= first - UNRELIABLE_NEIGHBOUR_S) & (everything.t_s <= last + UNRELIABLE_NEIGHBOUR_S)]
        near_start = around[around.t_s <= first + 0.2].touches_edge.any()
        near_end = around[around.t_s >= last - 0.2].touches_edge.any()
        tracklets.append(Tracklet(next_id, camera, piece.t_s.to_numpy(), xy, cov, None, bool(near_start), bool(near_end)))
        piece["X"], piece["Y"] = xy[:, 0], xy[:, 1]
        unknown = around[around.source == "unreliable"].copy()
        members = pd.concat([piece, unknown], ignore_index=True) if len(unknown) else piece
        members["tracklet"] = next_id
        assigned.append(members)
        next_id += 1
    _set_embeddings(tracklets, appearance)
    table = pd.concat(assigned, ignore_index=True) if assigned else pd.DataFrame()
    return tracklets, table, pd.DataFrame(dropped)


def _set_embeddings(tracklets: list[Tracklet], appearance: dict[int, np.ndarray]) -> None:
    """Вектор треклета: среднее по снимкам после вычитания среднего вектора камеры.

    В векторе есть свет и ракурс камеры, из-за них разные люди на одной камере кажутся похожими.
    Среднее считается только по настоящим людям: отражения и экраны фильтр уже убрал.
    """
    by_camera: dict[str, list[np.ndarray]] = {}
    for tracklet in tracklets:
        if tracklet.id in appearance:
            by_camera.setdefault(tracklet.camera, []).append(appearance[tracklet.id])
    camera_mean = {camera: np.vstack(parts).mean(axis=0) for camera, parts in by_camera.items()}
    for tracklet in tracklets:
        if tracklet.id in appearance:
            centered = appearance[tracklet.id] - camera_mean[tracklet.camera]
            centered /= np.maximum(np.linalg.norm(centered, axis=1, keepdims=True), 1e-12)
            mean = centered.mean(axis=0)
            tracklet.embedding = mean / max(float(np.linalg.norm(mean)), 1e-12)


def _dropped_row(camera: str, raw_track: int, piece: pd.DataFrame, reason: str) -> dict:
    return {
        "camera": camera, "raw_track": raw_track, "t_start": piece.t_s.iloc[0], "t_end": piece.t_s.iloc[-1],
        "frame_start": int(piece.frame.min()), "frame_end": int(piece.frame.max()),
        "observations": len(piece), "conf_max": round(float(piece.conf.max()), 2),
        "margin": round(float(piece.margin.median()), 2), "height": round(float(piece.person_height.median()), 2),
        "reason": reason,
    }


def _locate_file(path: Path, camera: Camera, plan: Plan, cache: Path) -> None:
    locate_observations(pd.read_parquet(path), camera, plan).to_parquet(cache, index=False)


def filter_config(path: Path = FEATURES) -> FilterConfig:
    features = yaml.safe_load(path.read_text())
    mirrors = tuple((camera, tuple(region)) for camera, regions in (features.get("mirrors") or {}).items() for region in regions)
    return FilterConfig(mirrors=mirrors)


def load_perception(directory: Path, cameras: dict[str, Camera], plan: Plan) -> tuple[pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Наблюдения всех камер на плане. Положения считаются раз и лежат в кэше, а время всегда берётся из текущих часов."""
    clocks = yaml.safe_load((CAMERAS_CONFIG.parent / "clocks.yaml").read_text())
    names = [name for name in cameras if (directory / f"{name}.parquet").exists()]
    caches = {name: directory / f"{name}_located_v{LOCATE_VERSION}.parquet" for name in names}
    stale = [
        name for name in names
        if not caches[name].exists()
        or caches[name].stat().st_mtime < max(source.stat().st_mtime for source in (directory / f"{name}.parquet", CAMERAS_CONFIG, FEATURES))
    ]
    if stale:
        with ProcessPoolExecutor(len(stale)) as pool:
            for job in [pool.submit(_locate_file, directory / f"{name}.parquet", cameras[name], plan, caches[name]) for name in stale]:
                job.result()

    tables, embeddings = [], {}
    for name in names:
        located = pd.read_parquet(caches[name])
        located["t_s"] = clocks[name]["clock_at_frame_0"] + located.frame / clocks[name]["fps"]
        tables.append(located)
        stored = np.load(directory / f"{name}_emb.npz")
        embeddings[name] = (stored["keys"], stored["vectors"])
    return pd.concat(tables, ignore_index=True), embeddings
