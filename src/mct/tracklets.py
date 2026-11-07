"""Из наблюдений детектора в треклеты на плане: положение, погрешность, резка и фильтр ложных детекций."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
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
OWN_HEIGHT_QUANTILE = 0.75  # верхняя часть роста: ноги закрыты только в части кадров
OWN_HEIGHT_MIN_ROWS = 5
OWN_HEIGHT_RANGE = (1.65, 2.0)  # снизу 1.65: у куска, где ноги закрыты всё время, верхняя часть роста тоже занижена
OWN_HEIGHT_SIGMA_M = 0.06
HIDDEN_FEET_MARGIN_M = 0.2
HIDDEN_FEET_FROM_M = 15.0  # ближе низ рамки уходит мало, а положение по макушке само по себе хуже
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


SCREEN_REASON = "экран: закрытая зона"


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
    # Экраны: закрытые зоны кадра. Всё, что лежит внутри целиком, не человек, а человек перед экраном выходит за край
    screens: tuple[tuple[str, tuple[float, float, float, float]], ...] = ()
    # Трекер перескочил на другого человека. Смена ракурса тоже меняет внешность, поэтому нужны доказательства с двух сторон
    jump_max_cross: float = 0.4  # половины куска настолько непохожи
    jump_min_side: int = 2  # посекундных снимков с каждой стороны
    jump_window: int = 5  # сколько снимков у точки разреза сравнивать
    jump_min_match: float = 0.3  # половина похожа на кусок рядом
    jump_min_lead: float = 0.35  # и заметно сильнее, чем другая половина
    jump_near_m: float = 1.5
    jump_near_s: float = 1.5
    jump_apart_m: float = 1.5  # одновременно настолько далеко: это заведомо два человека
    swap_min_gain: float = 0.8  # двое обменялись номерами: начало одного похоже на конец другого сильнее, чем на свой конец
    swap_min_match: float = 0.5  # оба перекрёстных сравнения высокие: общий переход в другой свет даёт их ниже
    swap_touch_s: float = 1.0  # и рамки в этот момент соприкасались
    jump_passes: int = 2


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


def relocate_hidden_feet(located: pd.DataFrame, camera: Camera, margin_m: float = HIDDEN_FEET_MARGIN_M, near_m: float = HIDDEN_FEET_FROM_M) -> pd.DataFrame:
    """Ноги закрыты другим человеком: рост в кадре заметно ниже обычного роста этого номера трекера.

    Вдали низ рамки уходит на метры, поэтому такие точки считаем по макушке, но уже с собственным ростом человека, а не со средним.
    """
    if located.empty:
        return located
    located = located.copy()
    clean = located[(located.source == "foot") & ~located.touches_edge]
    counts = clean.groupby("raw_track").size()
    own = clean.groupby("raw_track").person_height.quantile(OWN_HEIGHT_QUANTILE)[counts >= OWN_HEIGHT_MIN_ROWS].clip(*OWN_HEIGHT_RANGE)
    heights = located.raw_track.map(own)
    far = np.hypot(located.foot_x - camera.x, located.foot_y - camera.y) >= near_m
    chosen = far & heights.notna() & ((located.source == "head") | ((located.source == "foot") & (located.person_height < heights - margin_m)))
    if not chosen.any():
        return located
    rows = located[chosen]
    head = rows[["head_u", "head_v"]].to_numpy()
    foot = rows[["foot_u", "foot_v"]].to_numpy()
    height = heights[chosen].to_numpy()
    span = np.maximum(np.linalg.norm(head - foot, axis=1, keepdims=True), 1e-6)
    along = (head - foot) / span
    across = np.column_stack([-along[:, 1], along[:, 0]])
    floor = camera.floors_from_heads(head, height)
    a = camera.floors_from_heads(head + along * (4.0 + 0.03 * span), height) - floor
    b = camera.floors_from_heads(head + across * 3.0, height) - floor
    c = camera.floors_from_heads(head, height + OWN_HEIGHT_SIGMA_M) - floor
    near = np.hypot(floor[:, 0] - camera.x, floor[:, 1] - camera.y) <= MAX_HEAD_DISTANCE_M
    index = rows.index[near]
    located.loc[index, "X"], located.loc[index, "Y"] = floor[near, 0], floor[near, 1]
    located.loc[index, "cov_xx"] = (a[:, 0] ** 2 + b[:, 0] ** 2 + c[:, 0] ** 2)[near]
    located.loc[index, "cov_xy"] = (a[:, 0] * a[:, 1] + b[:, 0] * b[:, 1] + c[:, 0] * c[:, 1])[near]
    located.loc[index, "cov_yy"] = (a[:, 1] ** 2 + b[:, 1] ** 2 + c[:, 1] ** 2)[near]
    located.loc[index, "source"] = "head"
    return located


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
    if any(camera == piece.camera.iloc[0] and inside_share(box, region) >= cfg.mirror_inside_share for camera, region in cfg.screens):
        return SCREEN_REASON
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
    snapshots: dict[int, tuple[np.ndarray, np.ndarray]] = {}
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
            keys, vectors = embeddings[camera][:2]
            if len(vectors):
                chosen = (keys[:, 1] == raw_track) & np.isin(keys[:, 0], piece.frame.to_numpy())
                if chosen.any():
                    appearance[next_id] = vectors[chosen]
                    snapshots[next_id] = keys[chosen, 0], np.interp(keys[chosen, 0], piece.frame.to_numpy(), piece.t_s.to_numpy())
        first, last = piece.t_s.iloc[0], piece.t_s.iloc[-1]
        around = everything[(everything.t_s >= first - UNRELIABLE_NEIGHBOUR_S) & (everything.t_s <= last + UNRELIABLE_NEIGHBOUR_S)]
        near_start = around[around.t_s <= first + 0.2].touches_edge.any()
        near_end = around[around.t_s >= last - 0.2].touches_edge.any()
        boxes = piece[["x1", "y1", "x2", "y2"]].to_numpy(float)
        tracklets.append(Tracklet(next_id, camera, piece.t_s.to_numpy(), xy, cov, None, bool(near_start), bool(near_end), boxes))
        piece["X"], piece["Y"] = xy[:, 0], xy[:, 1]
        unknown = around[around.source == "unreliable"].copy()
        members = pd.concat([piece, unknown], ignore_index=True) if len(unknown) else piece
        members["tracklet"] = next_id
        assigned.append(members)
        next_id += 1
    _set_embeddings(tracklets, appearance)
    table = pd.concat(assigned, ignore_index=True) if assigned else pd.DataFrame()
    tracklets, table = _split_identity_jumps(tracklets, table, appearance, snapshots, cfg)
    _set_dense_embeddings(tracklets, table, {camera: value[2] for camera, value in embeddings.items() if len(value) > 2 and value[2] is not None})
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


def _set_dense_embeddings(tracklets: list[Tracklet], table: pd.DataFrame, dense: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    """Вектор куска по снимкам чаще раза в секунду: один неудачный кадр меньше решает, особенно у коротких кусков.

    Разрезы кусков считаются до этого, по снимкам раз в секунду: их окна заданы в снимках.
    """
    by_id = {x.id: x for x in tracklets}
    for camera, (keys, vectors) in dense.items():
        rows = pd.DataFrame(keys, columns=["frame", "raw_track"])
        mine = table[table.camera == camera][["frame", "raw_track", "tracklet"]].drop_duplicates(["frame", "raw_track"])
        owners = rows.merge(mine, on=["frame", "raw_track"], how="left").tracklet.to_numpy()
        real = ~np.isnan(owners)
        if not real.any():
            continue
        centered = vectors[real] - vectors[real].mean(axis=0)
        centered /= np.maximum(np.linalg.norm(centered, axis=1, keepdims=True), 1e-12)
        owners = owners[real]
        for owner in np.unique(owners):
            by_id[int(owner)].embedding = _unit(centered[owners == owner])


def _unit(vectors: np.ndarray) -> np.ndarray:
    mean = vectors.mean(axis=0)
    return mean / max(float(np.linalg.norm(mean)), 1e-12)


def _plan_distance(x: Tracklet, start: float, end: float, y: Tracklet, near_s: float) -> float | None:
    """Наименьшее расстояние на плане от участка куска до другого куска; None, если они далеко по времени."""
    low, high = max(start, y.start), min(end, y.end)
    if high >= low:
        times = np.linspace(low, high, max(int((high - low) / 0.2) + 1, 2))
        return float(np.min(np.hypot(*(x.at(times)[0] - y.at(times)[0]).T)))
    if 0 < start - y.end <= near_s:
        return float(np.hypot(*(x.at(np.array([start]))[0][0] - y.at(np.array([y.end]))[0][0])))
    if 0 < y.start - end <= near_s:
        return float(np.hypot(*(x.at(np.array([end]))[0][0] - y.at(np.array([y.start]))[0][0])))
    return None


def _surely_different(a: Tracklet, b: Tracklet, apart_m: float) -> bool:
    """Заведомо два человека: одновременно на одной камере или одновременно далеко друг от друга."""
    low, high = max(a.start, b.start), min(a.end, b.end)
    if high - low < 0.4:
        return False
    if a.camera == b.camera:
        return True
    times = np.linspace(low, high, max(int((high - low) / 0.2) + 1, 2))
    return float(np.median(np.hypot(*(a.at(times)[0] - b.at(times)[0]).T))) > apart_m


def _jump_strength(x: Tracklet, k: int, times: np.ndarray, vectors: np.ndarray, others: list[Tracklet], cfg: FilterConfig) -> float | None:
    """Насколько точка k похожа на перескок: у половин разная внешность и у каждой свой продолжатель рядом."""
    before, after = vectors[max(0, k - cfg.jump_window):k], vectors[k:k + cfg.jump_window]
    left, right = _unit(before), _unit(after)
    cross = float(left @ right)
    if cross > cfg.jump_max_cross:
        return None
    split = float(times[k])
    start, end = max(x.start, float(times[max(0, k - cfg.jump_window)])), min(x.end, float(times[min(len(times) - 1, k + cfg.jump_window - 1)]))
    for_left, for_right = [], []
    for y in others:
        if y is x or y.embedding is None or y.end < x.start - cfg.jump_near_s or y.start > x.end + cfg.jump_near_s:
            continue
        to_left, to_right = float(left @ y.embedding), float(right @ y.embedding)
        lead = to_left - to_right
        if to_left >= cfg.jump_min_match and lead >= cfg.jump_min_lead:
            distance = _plan_distance(x, start, split, y, cfg.jump_near_s)
            if distance is not None and distance <= cfg.jump_near_m:
                for_left.append((y, lead))
        if to_right >= cfg.jump_min_match and -lead >= cfg.jump_min_lead:
            distance = _plan_distance(x, split, end, y, cfg.jump_near_s)
            if distance is not None and distance <= cfg.jump_near_m:
                for_right.append((y, -lead))
    leads = [a[1] + b[1] for a in for_left for b in for_right if a[0] is not b[0] and _surely_different(a[0], b[0], cfg.jump_apart_m)]
    return max(leads) - cross if leads else None


def find_identity_jumps(tracklets: list[Tracklet], seconds: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]], cfg: FilterConfig) -> dict[int, int]:
    """Кусок и номер посекундного снимка, с которого на нём уже другой человек."""
    jumps = {}
    for x in tracklets:
        if x.id not in seconds or x.embedding is None:
            continue
        _, times, vectors = seconds[x.id]
        scored = [(_jump_strength(x, k, times, vectors, tracklets, cfg), k) for k in range(cfg.jump_min_side, len(vectors) - cfg.jump_min_side + 1)]
        scored = [(strength, k) for strength, k in scored if strength is not None]
        if scored:
            jumps[x.id] = max(scored)[1]
    return jumps


def _boxes_touch(a: Tracklet, b: Tracklet, time: float, within_s: float) -> bool:
    """Рамки двух кусков одной камеры пересекаются хотя бы раз рядом с этим моментом."""
    if a.boxes is None or b.boxes is None:
        return False
    for t in np.arange(time - within_s, time + within_s + 1e-9, 0.2):
        if a.start <= t <= a.end and b.start <= t <= b.end:
            p, q = a.boxes[int(np.abs(a.t - t).argmin())], b.boxes[int(np.abs(b.t - t).argmin())]
            if p[0] < q[2] and q[0] < p[2] and p[1] < q[3] and q[1] < p[3]:
                return True
    return False


def find_tail_swaps(tracklets: list[Tracklet], seconds: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]], cfg: FilterConfig) -> dict[int, int]:
    """Двое обменялись номерами в трекере одной камеры: кусок и снимок, с которого на нём другой человек.

    Смаз и свет портят одно сравнение, но не все четыре сразу, поэтому такой обмен почти не бывает ложным.
    """
    swaps: dict[int, int] = {}
    best: dict[int, float] = {}
    ready = [x for x in tracklets if x.id in seconds]
    for index, a in enumerate(ready):
        for b in ready[index + 1:]:
            if a.camera != b.camera or min(a.end, b.end) - max(a.start, b.start) < 2.0:
                continue
            times_a, vectors_a = seconds[a.id][1:]
            times_b, vectors_b = seconds[b.id][1:]
            low, high = max(a.start, b.start), min(a.end, b.end)
            found = None
            for t in np.unique(np.concatenate([times_a[(times_a > low) & (times_a < high)], times_b[(times_b > low) & (times_b < high)]])):
                k_a, k_b = int(np.searchsorted(times_a, t)), int(np.searchsorted(times_b, t))
                sides = vectors_a[max(0, k_a - cfg.jump_window):k_a], vectors_a[k_a:k_a + cfg.jump_window], vectors_b[max(0, k_b - cfg.jump_window):k_b], vectors_b[k_b:k_b + cfg.jump_window]
                if min(len(side) for side in sides) < cfg.jump_min_side:
                    continue
                left_a, right_a, left_b, right_b = [_unit(side) for side in sides]
                cross_a, cross_b = float(left_a @ right_b), float(left_b @ right_a)
                gain = cross_a + cross_b - float(left_a @ right_a) - float(left_b @ right_b)
                if gain >= cfg.swap_min_gain and min(cross_a, cross_b) >= cfg.swap_min_match and (found is None or gain > found[0]):
                    found = gain, k_a, k_b, float(t)
            if found is None or not _boxes_touch(a, b, found[3], cfg.swap_touch_s):
                continue
            for x, k in ((a, found[1]), (b, found[2])):
                if found[0] > best.get(x.id, 0.0):
                    best[x.id], swaps[x.id] = found[0], k
    return swaps


def _split_identity_jumps(
    tracklets: list[Tracklet],
    table: pd.DataFrame,
    appearance: dict[int, np.ndarray],
    snapshots: dict[int, tuple[np.ndarray, np.ndarray]],
    cfg: FilterConfig,
) -> tuple[list[Tracklet], pd.DataFrame]:
    """Режет куски там, где трекер перескочил на другого человека: дальше половины склеиваются обычными правилами."""
    if not appearance:
        return tracklets, table
    means: dict[str, np.ndarray] = {}
    for camera in {x.camera for x in tracklets if x.id in appearance}:
        means[camera] = np.vstack([appearance[x.id] for x in tracklets if x.camera == camera and x.id in appearance]).mean(axis=0)
    seconds = {}
    for x in tracklets:
        if x.id in appearance:
            frames, times = snapshots[x.id]
            order = np.argsort(frames)
            centered = appearance[x.id][order] - means[x.camera]
            seconds[x.id] = frames[order], times[order], centered / np.maximum(np.linalg.norm(centered, axis=1, keepdims=True), 1e-12)
    table = table.copy()
    next_id = max(x.id for x in tracklets) + 1
    for _ in range(cfg.jump_passes):  # чистые половины становятся доказательством для кусков, которые перескочили вместе с ними
        jumps = {**find_identity_jumps(tracklets, seconds, cfg), **find_tail_swaps(tracklets, seconds, cfg)}
        result = []
        changed = False
        for x in tracklets:
            frames, times, vectors = seconds.get(x.id, (None, None, None))
            cut = int(np.searchsorted(x.t, times[jumps[x.id]])) if x.id in jumps else 0
            if not 0 < cut < len(x.t):
                result.append(x)
                continue
            k = jumps[x.id]
            boxes_left = None if x.boxes is None else x.boxes[:cut]
            boxes_right = None if x.boxes is None else x.boxes[cut:]
            result.append(replace(x, t=x.t[:cut], xy=x.xy[:cut], cov=x.cov[:cut], boxes=boxes_left, embedding=_unit(vectors[:k]), ends_at_edge=False))
            result.append(replace(x, id=next_id, t=x.t[cut:], xy=x.xy[cut:], cov=x.cov[cut:], boxes=boxes_right, embedding=_unit(vectors[k:]), starts_at_edge=False))
            table.loc[(table.tracklet == x.id) & (table.frame >= frames[k]), "tracklet"] = next_id
            seconds[x.id], seconds[next_id] = (frames[:k], times[:k], vectors[:k]), (frames[k:], times[k:], vectors[k:])
            next_id += 1
            changed = True
        tracklets = result
        if not changed:
            break
    return tracklets, table


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
    features = yaml.safe_load(path.read_text(encoding="utf-8"))
    def regions(name: str) -> tuple:
        return tuple((camera, tuple(region)) for camera, items in (features.get(name) or {}).items() for region in items)
    return FilterConfig(mirrors=regions("mirrors"), screens=regions("screens"))


def load_perception(directory: Path, cameras: dict[str, Camera], plan: Plan, feet_margin_m: float | None = HIDDEN_FEET_MARGIN_M, near_m: float = HIDDEN_FEET_FROM_M, use_dense: bool = True) -> tuple[pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Наблюдения всех камер на плане. Положения считаются раз и лежат в кэше, а время всегда берётся из текущих часов."""
    clocks = yaml.safe_load((CAMERAS_CONFIG.parent / "clocks.yaml").read_text(encoding="utf-8"))
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
        if feet_margin_m is not None:
            located = relocate_hidden_feet(located, cameras[name], feet_margin_m, near_m)
        if located.empty:
            continue
        located["t_s"] = clocks[name]["clock_at_frame_0"] + located.frame / clocks[name]["fps"]
        tables.append(located)
        stored = np.load(directory / f"{name}_emb.npz")
        dense_path = directory / f"{name}_emb_dense.npz"
        dense = None
        if use_dense and dense_path.exists():
            frequent = np.load(dense_path)
            dense = frequent["keys"], frequent["vectors"]
        embeddings[name] = (stored["keys"], stored["vectors"], dense)
    return pd.concat(tables, ignore_index=True), embeddings
