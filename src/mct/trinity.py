"""Trinity Check: склейка треклетов в сквозные треки по геометрии, времени и внешности.

Вход не зависит от видео: положения на плане в метрах, их погрешности и векторы внешности.
Порядок строгий: устойчивое перекрытие, затем короткий контакт, затем разрывы. Поэтому короткая
встреча не может отобрать человека у надёжной передачи.
"""
from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

STEP_S = 0.2  # шаг сравнения траекторий
RECORDING_END_MARGIN_S = 1.0  # человека видно до самого конца записи: куда он делся дальше, неизвестно
MAX_SAMPLE_GAP_S = 0.5  # дальше этого от ближайшего наблюдения положение считается неизвестным


@dataclass(frozen=True)
class TrinityConfig:
    k: float = 3.0  # допуск по расстоянию Махаланобиса
    min_inlier_share: float = 0.7
    conflict_k: float = 4.5  # выше этого перекрывающиеся треклеты одной группы противоречат друг другу
    conflict_min_overlap_s: float = 0.5
    same_camera_overlap_s: float = 0.5  # трекер камеры держит старый номер пару кадров после выдачи нового: это один человек
    sustained_s: float = 1.0
    brief_gap_s: float = 0.5
    extrapolation_s: float = 1.0
    extrapolation_sigma: float = 0.5  # м на секунду продления
    calibration_sigma: float = 0.35  # м: ошибка привязки и синхронизации сверх погрешности точки
    v_max: float = 3.0
    a_max: float = 4.0  # м/с²: разворот на месте за доли секунды невозможен
    direction_min_speed: float = 0.5
    direction_min_cos: float = 0.5
    score_margin: float = 0.05  # запас лучшего кандидата на уровнях B и C
    gap_tolerance_cap_m: float = 1.5  # вдали погрешность огромна, и без предела разрыв пропускал бы скачки на 12 м
    reach_speed: float = 2.2  # сколько человек реально проходит за разрыв; v_max остаётся пределом для запрета
    brief_max_m: float = 2.0  # короткому контакту нужно и малое расстояние в метрах, а не только с учётом погрешности
    adjacent_gap_s: float = 1.5  # почти стык: новый трек начался там же почти сразу
    adjacent_distance_m: float = 0.5
    yellow_d_margin: float = 1.5  # конкурент не дальше этого по Махаланобису неотличим по геометрии
    yellow_r_margin: float = 0.1  # и не хуже этого по внешности
    # Пороги сходства после вычитания среднего вектора камеры, измерены на записи: scripts/measure_reid.py
    r_veto: float = 0.05  # проходит 98% своих: ниже почти никогда не бывает один человек на двух камерах
    r_gap: float = 0.2  # проходит 84% своих и 6% чужих
    gap_veto: float = 0.15  # на разрыве нет общего времени, поэтому запрет строже: все явные ошибки на снимках были при R ниже
    r_hint: float = 0.6  # уровень целых треков: проходит 40% своих и 3% чужих
    hint_margin: float = 0.1
    min_track_s: float = 2.0
    contact_r: float = 0.7  # один объект на той же камере: рамки пересекаются и внешность совпадает, разворот ему не запрещён
    contact_gap_s: float = 0.5
    edge_override_r: float = 0.6  # короткий контакт не у края кадра допустим при таком сходстве
    share_override_r: float = 0.55  # меньшая доля совпавших точек допустима при таком сходстве
    share_override_min: float = 0.5
    door_gap_veto: float = 0.2  # стык у двери на одной камере: один вышел, другой вошёл, а трекер уже дал новый номер
    return_window_s: float = 60.0  # организаторы: длительное отсутствие, это больше минуты без человека на всех камерах
    r_return: float = 0.6  # вернувшегося можно узнать только по внешности: на снимках от этого порога пары верные
    return_margin: float = 0.1  # заведомо другой человек похож почти так же: это не решаем
    duplicate_iou: float = 0.45  # трекер завёл второй номер на того же человека: две почти одинаковые рамки одного кадра
    duplicate_r: float = 0.5
    duplicate_max_s: float = 3.0  # пара в обнимку тоже даёт похожие рамки и вырезки, но так долго: второй номер на человека быстро пропадает
    weight_g: float = 0.5
    weight_t: float = 0.3
    weight_r: float = 0.2


@dataclass
class Tracklet:
    id: int
    camera: str
    t: np.ndarray  # секунды общего времени, по возрастанию
    xy: np.ndarray  # N×2, метры на плане
    cov: np.ndarray  # N×2×2, погрешность положения
    embedding: np.ndarray | None = None
    starts_at_edge: bool = False
    ends_at_edge: bool = False
    boxes: np.ndarray | None = None  # N×4, рамки на кадре в порядке t

    @property
    def start(self) -> float:
        return float(self.t[0])

    @property
    def end(self) -> float:
        return float(self.t[-1])

    def at(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Положение, погрешность и признак, что рядом есть настоящее наблюдение."""
        xy = np.column_stack([np.interp(times, self.t, self.xy[:, 0]), np.interp(times, self.t, self.xy[:, 1])])
        right = np.clip(np.searchsorted(self.t, times), 0, len(self.t) - 1)
        left = np.maximum(right - 1, 0)
        nearest = np.where(np.abs(self.t[left] - times) <= np.abs(self.t[right] - times), left, right)
        return xy, self.cov[nearest], np.abs(self.t[nearest] - times) <= MAX_SAMPLE_GAP_S

    def state_near(self, time: float, window_s: float = 1.0) -> tuple[np.ndarray, np.ndarray | None]:
        """Положение и скорость по прямой через наблюдения около момента time."""
        mask = np.abs(self.t - time) <= window_s
        if mask.sum() < 3 or np.ptp(self.t[mask]) < 0.2:
            return self.at(np.array([time]))[0][0], None
        times = self.t[mask] - time
        slope_x, level_x = np.polyfit(times, self.xy[mask, 0], 1)
        slope_y, level_y = np.polyfit(times, self.xy[mask, 1], 1)
        return np.array([level_x, level_y]), np.array([slope_x, slope_y])


@dataclass(frozen=True)
class Zone:
    kind: str  # DOOR, OCCLUDER или EDGE_LINK
    x: float
    y: float
    r: float
    to_x: float | None = None  # EDGE_LINK: где человек должен появиться
    to_y: float | None = None
    to_r: float | None = None

    def contains(self, xy: np.ndarray) -> bool:
        return float(np.hypot(xy[0] - self.x, xy[1] - self.y)) <= self.r


@dataclass(frozen=True)
class GapPolicy:
    kind: str
    buffer_s: float
    reid: str  # required или no_conflict
    center: tuple[float, float] | None  # где должен появиться человек; None: от места потери
    radius: float | None  # None: сколько можно пройти за разрыв
    pause: str = "occluded"


@dataclass
class ZoneMap:
    zones: list[Zone] = field(default_factory=list)
    strict_blind: bool = False
    recording_end: dict[str, float] = field(default_factory=dict)  # когда кончается запись каждой камеры
    exits: dict[str, list[tuple[float, float, float, float]]] = field(default_factory=dict)  # выходы без двери на кадре: лестница
    door_buffer_s: float = 1.5
    stationary_buffer_s: float = 30.0
    occluder_buffer_s: float = 5.0
    link_buffer_s: float = 5.0
    edge_out_buffer_s: float = 1.5
    open_buffer_s: float = 2.0
    stationary_radius: float = 1.0
    edge_out_radius: float = 1.5
    stationary_speed: float = 0.2

    @property
    def max_buffer_s(self) -> float:
        return max(self.door_buffer_s, self.stationary_buffer_s, self.occluder_buffer_s, self.link_buffer_s,
                   self.edge_out_buffer_s, self.open_buffer_s)

    def policy(self, tracklet: Tracklet) -> GapPolicy:
        """Правила раздела 5: приоритет DOOR, затем стоял на месте, OCCLUDER, EDGE_LINK, EDGE_OUT, OPEN."""
        recent = tracklet.t >= tracklet.end - 0.5
        lost_at = np.median(tracklet.xy[recent], axis=0)
        by_kind = lambda kind: next((z for z in self.zones if z.kind == kind and z.contains(lost_at)), None)  # noqa: E731

        if door := by_kind("DOOR"):
            return GapPolicy("DOOR", self.door_buffer_s, "required", (door.x, door.y), door.r)
        if self._was_standing(tracklet):
            return GapPolicy("STATIONARY", self.stationary_buffer_s, "required", tuple(lost_at), self.stationary_radius)
        if occluder := by_kind("OCCLUDER"):
            return GapPolicy("OCCLUDER", self.occluder_buffer_s, "no_conflict", (occluder.x, occluder.y), 1.5 * occluder.r)
        if link := by_kind("EDGE_LINK"):
            buffer_s = 0.0 if self.strict_blind else self.link_buffer_s
            return GapPolicy("EDGE_LINK", buffer_s, "required", (link.to_x, link.to_y), link.to_r, "blind_inferred")
        if tracklet.ends_at_edge:
            return GapPolicy("EDGE_OUT", self.edge_out_buffer_s, "required", tuple(lost_at), self.edge_out_radius)
        return GapPolicy("OPEN", self.open_buffer_s, "no_conflict", None, None)

    def closing_kind(self, tracklet: Tracklet) -> str:
        """Причина закрытия для показа. Вдали от камеры точка легко не попадает в зону двери, поэтому здесь учитывается ошибка положения. На склейку это не влияет."""
        kind = self.policy(tracklet).kind
        if kind != "OPEN":
            return kind
        recent = tracklet.t >= tracklet.end - 0.5
        lost_at = np.median(tracklet.xy[recent], axis=0)
        sigma = float(np.sqrt(np.median(np.trace(tracklet.cov[recent], axis1=1, axis2=2))))
        if any(zone.kind == "DOOR" and float(np.hypot(lost_at[0] - zone.x, lost_at[1] - zone.y)) <= zone.r + sigma for zone in self.zones):
            return "DOOR"
        if tracklet.boxes is not None and len(tracklet.boxes):
            # На лестнице положение на плане уходит за стену, поэтому выход размечен прямо на кадре
            x1, y1, x2, y2 = tracklet.boxes[-1]
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            if any(a <= center[0] <= c and b <= center[1] <= d for a, b, c, d in self.exits.get(tracklet.camera, [])):
                return "STAIRS"
        return kind

    def _was_standing(self, tracklet: Tracklet) -> bool:
        recent = tracklet.t >= tracklet.end - 2.0
        if np.ptp(tracklet.t[recent]) < 1.0:
            return False
        xy, t = tracklet.xy[recent], tracklet.t[recent]
        third = max(len(t) // 3, 1)
        moved = np.linalg.norm(np.median(xy[-third:], axis=0) - np.median(xy[:third], axis=0))
        return float(moved / np.ptp(t)) < self.stationary_speed


@dataclass
class Link:
    a: int
    b: int
    tier: str  # sustained, brief, gap или return
    t: float
    g_m: float  # расстояние между траекториями в метрах
    contact_s: float  # общее время; отрицательное значение: длина разрыва
    r: float | None
    score: float
    confidence: str = "green"
    pause: str | None = None
    d: float = 0.0  # медиана расстояния Махаланобиса


@dataclass
class Segment:
    kind: str  # observed, occluded или blind_inferred
    t_start: float
    t_end: float
    camera: str | None = None


@dataclass
class Hint:
    track_id: int
    r: float
    absent_s: float


@dataclass
class Track:
    track_id: int
    tracklets: list[int]
    cameras: list[str]
    t_start: float
    t_end: float
    links: list[Link]
    segments: list[Segment]
    close_zone: str
    hints: list[Hint] = field(default_factory=list)


@dataclass
class LinkResult:
    tracks: list[Track]
    dropped: list[int]
    checks: list[dict]


def _mahalanobis(delta: np.ndarray, cov: np.ndarray) -> np.ndarray:
    a, b, c = cov[:, 0, 0], cov[:, 0, 1], cov[:, 1, 1]
    return np.sqrt((c * delta[:, 0] ** 2 - 2 * b * delta[:, 0] * delta[:, 1] + a * delta[:, 1] ** 2) / (a * c - b * b))


def _similarity(a: Tracklet | None, b: Tracklet | None) -> float | None:
    if a is None or b is None or a.embedding is None or b.embedding is None:
        return None
    return float(a.embedding @ b.embedding)


def _disjoint(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _boxes_apart(a: Tracklet, b: Tracklet, time: float) -> bool:
    """Рамки одного кадра без общей площади: это два объекта, как бы близко ни легли точки на плане."""
    if a.boxes is None or b.boxes is None:
        return False
    return _disjoint(a.boxes[int(np.abs(a.t - time).argmin())], b.boxes[int(np.abs(b.t - time).argmin())])


def _number_switch(cfg: TrinityConfig, last: Tracklet, first: Tracklet, gap: float) -> bool:
    """На кадре это похоже на смену номера у одного объекта: та же камера, почти без паузы, рамки пересекаются."""
    if last.camera != first.camera or gap > cfg.contact_gap_s or last.boxes is None or first.boxes is None:
        return False
    return not _disjoint(last.boxes[-1], first.boxes[0])


def _same_object(cfg: TrinityConfig, last: Tracklet, first: Tracklet, gap: float) -> bool:
    """Трекер сменил номер у того же объекта: к смене номера добавляется совпадение внешности."""
    r = _similarity(last, first)
    return _number_switch(cfg, last, first, gap) and r is not None and r >= cfg.contact_r


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def _duplicate(cfg: TrinityConfig, a: Tracklet, b: Tracklet) -> bool:
    """Два номера трекера на одном человеке: та же камера, недолго вместе, рамки почти совпадают, внешность совпадает."""
    if a.camera != b.camera or a.boxes is None or b.boxes is None:
        return False
    start, end = max(a.start, b.start), min(a.end, b.end)
    if end <= start or end - start > cfg.duplicate_max_s:
        return False
    r = _similarity(a, b)
    if r is None or r < cfg.duplicate_r:
        return False
    times = np.arange(start, end + 1e-9, STEP_S)
    return float(np.median([_box_iou(a.boxes[int(np.abs(a.t - t).argmin())], b.boxes[int(np.abs(b.t - t).argmin())]) for t in times])) >= cfg.duplicate_iou


def _score(cfg: TrinityConfig, g: float, t: float, r: float | None) -> float:
    return cfg.weight_g * g + cfg.weight_t * t + cfg.weight_r * (0.5 if r is None else max(r, 0.0))


class _Linker:
    def __init__(self, tracklets: Iterable[Tracklet], cfg: TrinityConfig, zones: ZoneMap) -> None:
        self.cfg = cfg
        self.zones = zones
        self.tracklets = sorted(tracklets, key=lambda x: (x.start, x.camera, x.id))
        self.by_id = {x.id: x for x in self.tracklets}
        self.group_of = {x.id: x.id for x in self.tracklets}
        self.members = {x.id: [x] for x in self.tracklets}
        self.links: list[Link] = []
        self.checks: list[dict] = []
        self._calibration = np.eye(2) * cfg.calibration_sigma**2
        self._conflicts: dict[tuple[int, int], bool] = {}

    # Геометрия пары

    def _overlap_metrics(self, a: Tracklet, b: Tracklet) -> tuple[float, float, float] | None:
        """Медиана Махаланобиса, доля в допуске и медиана в метрах на общем участке."""
        start, end = max(a.start, b.start), min(a.end, b.end)
        times = np.arange(start, end + 1e-9, STEP_S)
        xy_a, cov_a, known_a = a.at(times)
        xy_b, cov_b, known_b = b.at(times)
        known = known_a & known_b
        if known.sum() < 3:
            return None
        delta = xy_a[known] - xy_b[known]
        distances = _mahalanobis(delta, cov_a[known] + cov_b[known] + self._calibration)
        return float(np.median(distances)), float((distances <= self.cfg.k).mean()), float(np.median(np.hypot(*delta.T)))

    def _extended(self, tracklet: Tracklet, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Положение с продлением за концы треклета по его скорости; погрешность растёт со временем."""
        cfg = self.cfg
        xy, cov, _ = tracklet.at(times)
        xy, cov = xy.copy(), cov.copy()
        for mask, edge in ((times > tracklet.end, tracklet.end), (times < tracklet.start, tracklet.start)):
            if not mask.any():
                continue
            level, velocity = tracklet.state_near(edge)
            dt = np.clip(times[mask] - edge, -cfg.extrapolation_s, cfg.extrapolation_s)
            sigma = cfg.extrapolation_sigma if velocity is not None else 2 * cfg.extrapolation_sigma
            xy[mask] = level + np.outer(dt, velocity if velocity is not None else np.zeros(2))
            cov[mask] += np.eye(2)[None] * ((sigma * dt) ** 2)[:, None, None]
        return xy, cov

    # Слияние групп

    def _pair_conflict(self, a: Tracklet, b: Tracklet) -> bool:
        key = (min(a.id, b.id), max(a.id, b.id))
        if key not in self._conflicts:
            self._conflicts[key] = self._compute_conflict(a, b)
        return self._conflicts[key]

    def _compute_conflict(self, a: Tracklet, b: Tracklet) -> bool:
        cfg = self.cfg
        overlap = min(a.end, b.end) - max(a.start, b.start)
        if a.camera == b.camera and overlap >= 0:
            if overlap > cfg.same_camera_overlap_s:
                return not _duplicate(cfg, a, b)
            middle = np.array([(max(a.start, b.start) + min(a.end, b.end)) / 2])
            if _boxes_apart(a, b, float(middle[0])):
                return True
            r = _similarity(a, b)
            if r is not None and r >= cfg.contact_r:
                return False  # рамки соприкасаются и внешность та же: это смена номера, а точки на плане расходятся из-за закрытых ног
            xy_a, cov_a, _ = a.at(middle)
            xy_b, cov_b, _ = b.at(middle)
            return float(_mahalanobis(xy_a - xy_b, cov_a + cov_b + self._calibration)[0]) > cfg.k
        if overlap >= cfg.conflict_min_overlap_s:
            metrics = self._overlap_metrics(a, b)
            return metrics is not None and metrics[0] > cfg.conflict_k
        if overlap > 0:
            return False
        first, second = (a, b) if a.end <= b.start else (b, a)
        gap = second.start - first.end
        # Рамки соприкасаются, а точки на плане далеко: вдали это скачок из-за закрытых ног, а не движение
        return not _number_switch(cfg, first, second, gap) and not self._reachable(first, second, gap)

    def _reachable(self, first: Tracklet, second: Tracklet, gap: float) -> bool:
        """Можно ли дойти за разрыв с учётом погрешности положения."""
        from_xy = first.state_near(first.end)[0]
        to_xy = second.state_near(second.start)[0]
        return float(np.hypot(*(to_xy - from_xy))) <= self.cfg.v_max * gap + self._tolerance(first, second)

    def _tolerance(self, first: Tracklet, second: Tracklet) -> float:
        total = first.cov[-1] + second.cov[0] + self._calibration
        return 2 * float(np.sqrt(np.linalg.eigvalsh(total)[-1]))

    def _can_merge(self, id_a: int, id_b: int) -> bool:
        group_a, group_b = self.group_of[id_a], self.group_of[id_b]
        if group_a == group_b:
            return False
        return not any(self._pair_conflict(a, b) for a in self.members[group_a] for b in self.members[group_b])

    def _merge(self, link: Link) -> None:
        keep, other = self.group_of[link.a], self.group_of[link.b]
        for tracklet in self.members[other]:
            self.group_of[tracklet.id] = keep
        self.members[keep] += self.members.pop(other)
        self.links.append(link)

    def _record(self, a: int, b: int, tier: str, passed: bool, reason: str, **values: float | None) -> None:
        self.checks.append({"a": a, "b": b, "tier": tier, "passed": passed, "reason": reason, **values})

    # Уровень A: устойчивое перекрытие

    def sustained(self) -> None:
        cfg = self.cfg
        passing: list[Link] = []
        items = self.tracklets
        for index, a in enumerate(items):
            for b in items[index + 1:]:
                if b.start > a.end - cfg.sustained_s:
                    break
                overlap = min(a.end, b.end) - b.start
                if a.camera == b.camera or overlap < cfg.sustained_s:
                    continue
                metrics = self._overlap_metrics(a, b)
                if metrics is None:
                    continue
                median_d, share, median_m = metrics
                r = _similarity(a, b)
                enough = share >= cfg.min_inlier_share or (r is not None and r >= cfg.share_override_r and share >= cfg.share_override_min)
                ok = median_d <= cfg.k and enough
                self._record(a.id, b.id, "sustained", ok, "" if ok else "геометрия", d=median_d, overlap=overlap, r=r, g_m=median_m)
                if ok:
                    g = max(0.0, 1 - median_d / cfg.k)
                    score = _score(cfg, g, min(overlap / 5.0, 1.0), r)
                    passing.append(Link(a.id, b.id, "sustained", b.start, median_m, overlap, r, score, d=median_d))

        by_tracklet: dict[int, list[Link]] = {}
        for link in passing:
            by_tracklet.setdefault(link.a, []).append(link)
            by_tracklet.setdefault(link.b, []).append(link)

        accepted = []
        for link in passing:
            rivals = [other for other in by_tracklet[link.a] + by_tracklet[link.b] if self._is_rival(link, other)]
            if rivals and link.r is not None and link.r < cfg.r_veto:
                self._record(link.a, link.b, "sustained", False, "внешность при конкуренте", r=link.r)
                continue
            if any(self._indistinguishable(link, other) for other in rivals):
                link.confidence = "yellow"
            accepted.append(link)

        for link in sorted(accepted, key=lambda x: (-x.score, x.a, x.b)):
            if self._can_merge(link.a, link.b):
                self._merge(link)

    def _indistinguishable(self, link: Link, other: Link) -> bool:
        if other.d > link.d + self.cfg.yellow_d_margin:
            return False
        return link.r is None or other.r is None or other.r >= link.r - self.cfg.yellow_r_margin

    def _is_rival(self, link: Link, other: Link) -> bool:
        """Конкурент: другой треклет той же камеры, что и партнёр, годится на то же место."""
        if other is link:
            return False
        shared = link.a if link.a in (other.a, other.b) else link.b
        partner = link.b if shared == link.a else link.a
        rival = other.b if other.a == shared else other.a
        return self.by_id[partner].camera == self.by_id[rival].camera

    # Уровень B: короткий контакт

    def brief(self) -> None:
        cfg = self.cfg
        candidates: list[Link] = []
        items = self.tracklets
        for index, first in enumerate(items):
            for second in items[index + 1:]:
                if second.start > first.end + cfg.brief_gap_s:
                    break
                if first.camera == second.camera or self.group_of[first.id] == self.group_of[second.id]:
                    continue
                overlap = min(first.end, second.end) - second.start
                if overlap >= cfg.sustained_s or not self._can_merge(first.id, second.id):
                    continue
                link = self._check_brief(first, second, overlap)
                if link is not None:
                    candidates.append(link)

        accepted = []
        for link in candidates:
            rivals = [other for other in candidates if self._competes_for_handoff(link, other)]
            if any(other.score > link.score - cfg.score_margin for other in rivals):
                self._record(link.a, link.b, "brief", False, "неоднозначно", r=link.r)
                continue
            required = cfg.r_gap if rivals else cfg.r_veto
            if link.r is not None and link.r < required or (rivals and link.r is None):
                self._record(link.a, link.b, "brief", False, "внешность", r=link.r)
                continue
            self._record(link.a, link.b, "brief", True, "", r=link.r, g_m=link.g_m, overlap=link.contact_s)
            accepted.append(link)

        for link in sorted(accepted, key=lambda x: (-x.score, x.a, x.b)):
            if self._can_merge(link.a, link.b):
                self._merge(link)

    def _check_fragment(self, long: Tracklet, fragment: Tracklet) -> Link | None:
        cfg = self.cfg
        xy, cov, known = long.at(fragment.t)
        if not known.any():
            return None
        delta = xy[known] - fragment.xy[known]
        distances = _mahalanobis(delta, cov[known] + fragment.cov[known] + self._calibration)
        median_d = float(np.median(distances))
        if median_d > cfg.k or float(np.median(np.hypot(*delta.T))) > cfg.brief_max_m:
            self._record(long.id, fragment.id, "brief", False, "геометрия обломка", d=median_d)
            return None
        r = _similarity(long, fragment)
        contact = fragment.end - fragment.start
        return Link(long.id, fragment.id, "brief", fragment.start, float(np.median(np.hypot(*delta.T))), contact, r,
                    _score(cfg, max(0.0, 1 - median_d / cfg.k), 0.1, r), d=median_d)

    def _competes_for_handoff(self, link: Link, other: Link) -> bool:
        """Тот же предшественник или тот же преемник, но другой человек, то есть другая группа."""
        if other is link:
            return False
        if other.a == link.a:
            return self.group_of[other.b] != self.group_of[link.b]
        if other.b == link.b:
            return self.group_of[other.a] != self.group_of[link.a]
        return False

    def _check_brief(self, first: Tracklet, second: Tracklet, overlap: float) -> Link | None:
        cfg = self.cfg
        # Обломок: короткая детекция целиком внутри времени другого треклета. Край здесь не нужен,
        # потому что сравнивать можно наблюдаемые положения, а не продлённые
        if second.end <= first.end:
            return self._check_fragment(first, second)
        similar = _similarity(first, second)
        if not (first.ends_at_edge or second.starts_at_edge or (similar is not None and similar >= cfg.edge_override_r)):
            self._record(first.id, second.id, "brief", False, "край", overlap=overlap)
            return None

        contact = (second.start + min(first.end, second.end)) / 2
        before = first.state_near(min(first.end, contact))[1]
        after = second.state_near(second.start)[1]
        if before is not None and after is not None:
            speeds = np.linalg.norm(before), np.linalg.norm(after)
            if min(speeds) > cfg.direction_min_speed and (before @ after) / (speeds[0] * speeds[1]) < cfg.direction_min_cos:
                self._record(first.id, second.id, "brief", False, "направление", overlap=overlap)
                return None

        times = np.arange(contact - cfg.extrapolation_s, contact + cfg.extrapolation_s + 1e-9, STEP_S)
        xy_a, cov_a = self._extended(first, times)
        xy_b, cov_b = self._extended(second, times)
        delta = xy_a - xy_b
        median_d = float(np.median(_mahalanobis(delta, cov_a + cov_b + self._calibration)))
        if median_d > cfg.k or float(np.median(np.hypot(*delta.T))) > cfg.brief_max_m:
            self._record(first.id, second.id, "brief", False, "геометрия", d=median_d, overlap=overlap)
            return None

        r = _similarity(first, second)
        g = max(0.0, 1 - median_d / cfg.k)
        t = 0.2 * max(0.0, 1 + min(overlap, 0.0) / cfg.brief_gap_s)
        return Link(first.id, second.id, "brief", contact, float(np.median(np.hypot(*delta.T))), overlap, r,
                    _score(cfg, g, t, r), pause="blind_inferred" if overlap < 0 else None)

    # Уровень C: разрывы между группами

    # Тот же объект на одной камере: сравниваются сами куски, а не концы всего трека

    def contacts(self) -> None:
        cfg = self.cfg
        links: list[Link] = []
        items = self.tracklets
        for index, last in enumerate(items):
            for first in items[index + 1:]:
                if first.start > last.end + cfg.contact_gap_s:
                    break
                if first.camera != last.camera or first.end <= last.end or self.group_of[first.id] == self.group_of[last.id]:
                    continue
                gap = max(first.start - last.end, 0.0)
                if _same_object(cfg, last, first, gap):
                    r = _similarity(last, first)
                    distance = float(np.hypot(*(first.state_near(first.start)[0] - last.state_near(last.end)[0])))
                    links.append(Link(last.id, first.id, "gap", last.end, distance, -gap, r, _score(cfg, 1.0, 0.2, r), pause=self.zones.policy(last).pause))
        for index, a in enumerate(items):
            for b in items[index + 1:]:
                if b.start >= a.end:
                    break
                if self.group_of[a.id] != self.group_of[b.id] and _duplicate(cfg, a, b):
                    r = _similarity(a, b)
                    links.append(Link(a.id, b.id, "gap", b.start, 0.0, min(a.end, b.end) - b.start, r, _score(cfg, 1.0, 0.2, r), pause="occluded"))
        for link in sorted(links, key=lambda x: (-x.score, x.a, x.b)):
            if self._can_merge(link.a, link.b):
                self._record(link.a, link.b, "gap", True, "", r=link.r, g_m=link.g_m, overlap=link.contact_s)
                self._merge(link)

    # Возврат: человека нет ни на одной камере не дольше минуты

    def returns(self) -> None:
        while not self.zones.strict_blind and self._return_round():
            pass

    def _return_round(self) -> bool:
        cfg = self.cfg
        vectors = {}
        for group, members in self.members.items():
            vector = _members_embedding(members)
            if vector is not None and _observed_time(members) >= cfg.min_track_s:
                vectors[group] = vector
        possible: dict[frozenset, Link] = {}
        for g1, g2 in combinations(vectors, 2):
            link = self._return_link(self.members[g1], self.members[g2])
            if link is not None and self._can_merge(link.a, link.b):
                possible[frozenset((g1, g2))] = link

        def similarity(pair: frozenset) -> float:
            first, second = tuple(pair)
            return float(vectors[first] @ vectors[second])

        def overlaps(g1: int, g2: int) -> bool:
            return any(a.start < b.end and b.start < a.end for a in self.members[g1] for b in self.members[g2])

        def has_rival(group: int, partner: int, r: float) -> bool:
            # Группа похожа на другого человека почти так же, а он был виден одновременно с партнёром
            return any(
                other != partner and other in vectors and frozenset((group, other)) in possible
                and overlaps(other, partner) and float(vectors[group] @ vectors[other]) >= r - cfg.return_margin
                for other in vectors if other != group
            )

        for pair in sorted(possible, key=lambda p: (-similarity(p), min(p), max(p))):
            r = similarity(pair)
            if r < cfg.r_return:
                return False
            g1, g2 = tuple(pair)
            if has_rival(g1, g2, r) or has_rival(g2, g1, r):
                self._record(possible[pair].a, possible[pair].b, "return", False, "неоднозначно", r=r)
                continue
            link = possible[pair]
            link.r = r
            link.score = _score(cfg, 1.0, 0.0, r)
            self._record(link.a, link.b, "return", True, "", r=r, g_m=link.g_m, overlap=link.contact_s)
            self._merge(link)
            return True
        return False

    def _return_link(self, first_group: list[Tracklet], second_group: list[Tracklet]) -> Link | None:
        """Связь через паузу, если группы ни разу не видны одновременно и пауза не дольше минуты."""
        cfg = self.cfg
        items = sorted([(x.start, x.id, x, 0) for x in first_group] + [(x.start, x.id, x, 1) for x in second_group], key=lambda item: item[:2])
        latest: list[Tracklet | None] = [None, None]
        link = None
        for _, _, x, side in items:
            other = latest[1 - side]
            if other is not None:
                pause = x.start - other.end
                if pause <= 0:
                    return None
                own = latest[side]
                if (own is None or own.end < other.end) and link is None:
                    distance = float(np.hypot(*(x.xy[0] - other.xy[-1])))
                    if pause > cfg.return_window_s or distance > cfg.v_max * pause:
                        return None
                    link = Link(other.id, x.id, "return", other.end, distance, -pause, None, 0.0, pause="blind_inferred")
                elif (own is None or own.end < other.end) and (pause > cfg.return_window_s or float(np.hypot(*(x.xy[0] - other.xy[-1]))) > cfg.v_max * pause):
                    return None
            if latest[side] is None or x.end > latest[side].end:
                latest[side] = x
        return link

    def gaps(self) -> None:
        for _ in range(20):
            merged = self._gap_round()
            if not merged:
                return

    def _gap_round(self) -> int:
        cfg = self.cfg
        groups = sorted(self.members, key=lambda g: min(x.start for x in self.members[g]))
        ends = {g: max(self.members[g], key=lambda x: (x.end, x.id)) for g in groups}
        starts = {g: min(self.members[g], key=lambda x: (x.start, x.id)) for g in groups}

        candidates: list[tuple[int, int, Link]] = []
        for g1 in groups:
            last = ends[g1]
            policy = self.zones.policy(last)
            for g2 in groups:
                if g1 == g2:
                    continue
                first = starts[g2]
                gap = first.start - last.end
                if gap < -cfg.brief_gap_s or gap > policy.buffer_s:
                    continue
                link = self._check_gap(last, first, max(gap, 0.0), policy)
                if link is not None and self._can_merge(last.id, first.id):
                    candidates.append((g1, g2, link))

        merged = 0
        for g1, g2, link in sorted(candidates, key=lambda c: (-c[2].score, c[2].a, c[2].b)):
            rivals = [c[2] for c in candidates if c[2] is not link and (c[0] == g1 or c[1] == g2)]
            if any(other.score > link.score - cfg.score_margin for other in rivals):
                self._record(link.a, link.b, "gap", False, "неоднозначно", r=link.r)
                continue
            if self._can_merge(link.a, link.b):
                self._record(link.a, link.b, "gap", True, "", r=link.r, g_m=link.g_m, overlap=link.contact_s)
                self._merge(link)
                merged += 1
        return merged

    def _check_gap(self, last: Tracklet, first: Tracklet, gap: float, policy: GapPolicy) -> Link | None:
        cfg = self.cfg
        from_xy, before = last.state_near(last.end)
        to_xy, after = first.state_near(first.start)
        distance = float(np.hypot(*(to_xy - from_xy)))
        tolerance = min(self._tolerance(last, first), cfg.gap_tolerance_cap_m)
        if distance > cfg.v_max * gap + tolerance:
            self._record(last.id, first.id, "gap", False, "скорость", g_m=distance, overlap=-gap)
            return None
        if before is not None and after is not None and np.linalg.norm(after - before) / max(gap, 0.3) > cfg.a_max:
            self._record(last.id, first.id, "gap", False, "разворот", g_m=distance, overlap=-gap)
            return None

        if policy.center is None:
            reach = cfg.reach_speed * gap + 0.5 + tolerance
            miss = distance
        else:
            reach = policy.radius + tolerance
            miss = float(np.hypot(to_xy[0] - policy.center[0], to_xy[1] - policy.center[1]))
        if miss > reach:
            self._record(last.id, first.id, "gap", False, "место появления", g_m=miss, overlap=-gap)
            return None

        r = _similarity(last, first)
        # При почти стыке доказывают место и время. У коротких кусков мало снимков, поэтому внешность только запрещает
        adjacent = r is not None and gap <= cfg.adjacent_gap_s and distance <= cfg.adjacent_distance_m
        if policy.reid == "required" and not adjacent and (r is None or r < cfg.r_gap):
            self._record(last.id, first.id, "gap", False, "внешность", r=r, overlap=-gap)
            return None
        veto = cfg.door_gap_veto if policy.kind == "DOOR" and last.camera == first.camera else cfg.gap_veto
        if (policy.reid == "no_conflict" or adjacent) and r is not None and r < veto:
            self._record(last.id, first.id, "gap", False, "внешность", r=r, overlap=-gap)
            return None

        g = max(0.0, 1 - miss / max(reach, 1e-6))
        t = 0.2 * max(0.0, 1 - gap / max(policy.buffer_s, 1e-6))
        return Link(last.id, first.id, "gap", last.end, distance, -gap, r, _score(cfg, g, t, r), pause=policy.pause)

    # Итог

    def finish(self) -> LinkResult:
        cfg = self.cfg
        finished, dropped = [], []
        ordered = sorted(self.members.values(), key=lambda g: min((x.start, x.camera, x.id) for x in g))
        for members in ordered:
            start, end = min(x.start for x in members), max(x.end for x in members)
            if _observed_time(members) < cfg.min_track_s:
                dropped += sorted(x.id for x in members)
                continue
            finished.append(self._build_track(len(finished) + 1, members, start, end))
        _attach_hints(finished, self.by_id, cfg)
        return LinkResult(finished, sorted(dropped), self.checks)

    def _build_track(self, track_id: int, members: list[Tracklet], start: float, end: float) -> Track:
        ordered = sorted(members, key=lambda x: (x.start, x.camera, x.id))
        ids = {x.id for x in ordered}
        links = sorted((l for l in self.links if l.a in ids), key=lambda l: (l.t, l.a, l.b))
        segments = [Segment("observed", x.start, x.end, x.camera) for x in ordered]
        covered_until = ordered[0].end
        for tracklet in ordered[1:]:
            if tracklet.start > covered_until + 1e-9:
                pause = next((l.pause for l in links if l.pause and l.b == tracklet.id), None)
                segments.append(Segment(pause or "occluded", covered_until, tracklet.start))
            covered_until = max(covered_until, tracklet.end)
        segments.sort(key=lambda s: (s.t_start, s.t_end))
        last = max(ordered, key=lambda x: (x.end, x.id))
        cameras = sorted({x.camera for x in ordered})
        recording_ended = last.end >= self.zones.recording_end.get(last.camera, float("inf")) - RECORDING_END_MARGIN_S
        close_zone = "RECORDING_END" if recording_ended else self.zones.closing_kind(last)
        return Track(track_id, [x.id for x in ordered], cameras, start, end, links, segments, close_zone)


def _observed_time(members: list[Tracklet]) -> float:
    """Время, когда человека действительно видела хотя бы одна камера: цепочка обломков не становится треком."""
    intervals = sorted((x.start, x.end) for x in members)
    total, current_start, current_end = 0.0, *intervals[0]
    for start, end in intervals[1:]:
        if start > current_end:
            total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    return total + current_end - current_start


def _members_embedding(members: Iterable[Tracklet]) -> np.ndarray | None:
    vectors = [x.embedding for x in members if x.embedding is not None]
    if not vectors:
        return None
    mean = np.mean(vectors, axis=0)
    return mean / max(np.linalg.norm(mean), 1e-12)


def _track_embedding(track: Track, by_id: dict[int, Tracklet]) -> np.ndarray | None:
    return _members_embedding(by_id[i] for i in track.tracklets)


def _could_return(old: Track, new: Track, by_id: dict[int, Tracklet], cfg: TrinityConfig) -> bool:
    """Успел бы человек дойти от места ухода до места появления."""
    last = max((by_id[i] for i in old.tracklets), key=lambda x: x.end)
    first = min((by_id[i] for i in new.tracklets), key=lambda x: x.start)
    return float(np.hypot(*(first.xy[0] - last.xy[-1]))) <= cfg.v_max * (new.t_start - old.t_end)


def _attach_hints(tracks: list[Track], by_id: dict[int, Tracklet], cfg: TrinityConfig) -> None:
    """Призрак: новый трек похож на закрытый. Подсказка никогда не меняет ID и подсчёт."""
    embeddings = {track.track_id: _track_embedding(track, by_id) for track in tracks}
    pairs = []
    for new in tracks:
        if embeddings[new.track_id] is None:
            continue
        scored = []
        for old in tracks:
            if old.t_end < new.t_start and embeddings[old.track_id] is not None and _could_return(old, new, by_id, cfg):
                scored.append((float(embeddings[old.track_id] @ embeddings[new.track_id]), old))
        scored.sort(key=lambda x: (-x[0], x[1].track_id))
        if not scored or scored[0][0] < cfg.r_hint:
            continue
        if len(scored) > 1 and scored[0][0] - scored[1][0] < cfg.hint_margin:
            continue
        pairs.append((scored[0][0], new, scored[0][1]))

    used: set[int] = set()
    for r, new, old in sorted(pairs, key=lambda p: (-p[0], p[1].track_id)):
        if old.track_id not in used:
            used.add(old.track_id)
            new.hints.append(Hint(old.track_id, r, new.t_start - old.t_end))


def link_tracklets(tracklets: Iterable[Tracklet], cfg: TrinityConfig = TrinityConfig(), zones: ZoneMap | None = None) -> LinkResult:
    linker = _Linker(tracklets, cfg, zones or ZoneMap())
    linker.sustained()
    linker.brief()
    linker.contacts()
    linker.gaps()
    linker.returns()
    return linker.finish()


def check_invariants(result: LinkResult, tracklets: Iterable[Tracklet], cfg: TrinityConfig = TrinityConfig()) -> list[str]:
    """Проверки раздела 11: одна камера не видит трек дважды, скорость физична, у каждой паузы есть правило."""
    by_id = {x.id: x for x in tracklets}
    problems = []
    for track in result.tracks:
        members = sorted((by_id[i] for i in track.tracklets), key=lambda x: x.start)
        covered_until = members[0].end
        for piece in members[1:]:
            if piece.start - covered_until > cfg.return_window_s + 1e-9:
                problems.append(f"трек {track.track_id}: отсутствие дольше минуты между кусками")
            covered_until = max(covered_until, piece.end)
        for index, a in enumerate(members):
            for b in members[index + 1:]:
                overlap = min(a.end, b.end) - max(a.start, b.start)
                if a.camera == b.camera and overlap > cfg.same_camera_overlap_s and not _duplicate(cfg, a, b):
                    problems.append(f"трек {track.track_id}: треклеты {a.id} и {b.id} одной камеры пересекаются")
                elif a.camera == b.camera and overlap >= 0 and _boxes_apart(a, b, b.start + overlap / 2):
                    problems.append(f"трек {track.track_id}: треклеты {a.id} и {b.id} в разных рамках одного кадра")
        for segment in track.segments:
            if segment.kind != "observed" and not any(
                link.tier in ("gap", "brief", "return") and link.contact_s <= 0 for link in track.links
            ):
                problems.append(f"трек {track.track_id}: пауза без правила")
        tolerance = 3 * cfg.calibration_sigma
        for a, b in zip(members, members[1:]):
            gap = b.start - a.end
            if gap > 0 and not _number_switch(cfg, a, b, gap) and np.hypot(*(b.xy[0] - a.xy[-1])) > cfg.v_max * gap + tolerance + 3 * np.sqrt(
                max(np.linalg.eigvalsh(a.cov[-1])[-1], np.linalg.eigvalsh(b.cov[0])[-1])
            ):
                problems.append(f"трек {track.track_id}: скорость между {a.id} и {b.id} выше предела")
    ids = [i for track in result.tracks for i in track.tracklets] + result.dropped
    if len(ids) != len(set(ids)):
        problems.append("один треклет попал в два трека")
    known = {track.track_id: track for track in result.tracks}
    for track in result.tracks:
        for hint in track.hints:
            if hint.track_id not in known or known[hint.track_id].t_end >= track.t_start:
                problems.append(f"трек {track.track_id}: неверная подсказка")
    return problems
