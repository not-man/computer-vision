"""Сдвиги времени камер по людям: один человек на двух камерах в один момент стоит в одной точке плана.

Ошибка положения вдоль пути выглядит как сдвиг времени, но у встречных потоков меняет знак.
Поэтому сдвиг пары считается как среднее встречных направлений, и остаётся только время.
"""
from dataclasses import dataclass

import numpy as np

LAGS = np.arange(-2.5, 2.5001, 0.05)
MIN_OVERLAP_S = 1.0
MIN_SAMPLES = 8
MIN_SPEED = 0.6  # без движения сдвиг времени не определить
MIN_LAG_COVERAGE = 0.6
MAX_DISTANCE_M = 1.5
MIN_PER_DIRECTION = 2
ONE_DIRECTION_WEIGHT = 0.3  # одно направление не отделяет время от положения
MAX_PAIR_WEIGHT = 10


@dataclass(frozen=True)
class PairLag:
    camera_a: str
    camera_b: str
    direction: str  # большая ось движения и знак, например y+
    lag: float  # часы a опережают часы b на столько секунд
    distance: float  # расстояние в метрах при лучшем сдвиге
    distance_at_zero: float
    speed: float


def pair_lag(camera_a: str, t_a: np.ndarray, xy_a: np.ndarray, camera_b: str, t_b: np.ndarray, xy_b: np.ndarray) -> PairLag | None:
    """Сдвиг по одному человеку; None, если общего участка мало, человек стоит или минимум на краю поиска."""
    start, end = max(t_a[0], t_b[0]), min(t_a[-1], t_b[-1])
    if end - start < MIN_OVERLAP_S:
        return None
    t = t_a[(t_a >= start) & (t_a <= end)]
    if len(t) < MIN_SAMPLES:
        return None
    x, y = np.interp(t, t_a, xy_a[:, 0]), np.interp(t, t_a, xy_a[:, 1])
    dx, dy = x[-1] - x[0], y[-1] - y[0]
    speed = float(np.hypot(dx, dy) / (t[-1] - t[0]))
    if speed < MIN_SPEED:
        return None

    scores = np.full(len(LAGS), np.inf)
    for index, lag in enumerate(LAGS):
        inside = (t >= t_b[0] + lag) & (t <= t_b[-1] + lag)
        if inside.sum() >= MIN_LAG_COVERAGE * len(t):
            other_x = np.interp(t[inside], t_b + lag, xy_b[:, 0])
            other_y = np.interp(t[inside], t_b + lag, xy_b[:, 1])
            scores[index] = np.median(np.hypot(x[inside] - other_x, y[inside] - other_y))
    best = int(np.argmin(scores))
    if best in (0, len(LAGS) - 1) or scores[best] > MAX_DISTANCE_M:
        return None
    axis, delta = ("x", dx) if abs(dx) > abs(dy) else ("y", dy)
    zero = int(np.argmin(np.abs(LAGS)))
    return PairLag(camera_a, camera_b, f"{axis}{'+' if delta > 0 else '-'}", float(LAGS[best]), float(scores[best]), float(scores[zero]), speed)


def summarize_pair(lags: list[PairLag]) -> tuple[float, float]:
    """Сдвиг пары и вес: среднее встречных направлений, если они есть, иначе медиана с малым весом."""
    balanced, counts = [], []
    for axis in "xy":
        plus = [item.lag for item in lags if item.direction == f"{axis}+"]
        minus = [item.lag for item in lags if item.direction == f"{axis}-"]
        if len(plus) >= MIN_PER_DIRECTION and len(minus) >= MIN_PER_DIRECTION:
            balanced.append((np.median(plus) + np.median(minus)) / 2)
            counts.append(min(len(plus), len(minus)))
    if balanced:
        return float(np.average(balanced, weights=counts)), float(min(sum(counts), MAX_PAIR_WEIGHT))
    return float(np.median([item.lag for item in lags])), min(len(lags), MAX_PAIR_WEIGHT) * ONE_DIRECTION_WEIGHT


def camera_offsets(lags: list[PairLag], cameras: list[str]) -> tuple[dict[str, float], dict[tuple[str, str], float]]:
    """Насколько время каждой камеры опережает общее, сумма сдвигов равна нулю, и остатки по парам."""
    by_pair: dict[tuple[str, str], list[PairLag]] = {}
    for item in lags:
        by_pair.setdefault((item.camera_a, item.camera_b), []).append(item)
    summary = {pair: summarize_pair(items) for pair, items in sorted(by_pair.items())}

    rows, targets, weights = [], [], []
    for (a, b), (lag, weight) in summary.items():
        row = np.zeros(len(cameras))
        row[cameras.index(a)], row[cameras.index(b)] = 1.0, -1.0
        rows.append(row)
        targets.append(lag)
        weights.append(weight)
    rows.append(np.ones(len(cameras)))
    targets.append(0.0)
    weights.append(float(MAX_PAIR_WEIGHT))
    root = np.sqrt(np.array(weights))
    solution = np.linalg.lstsq(np.array(rows) * root[:, None], np.array(targets) * root, rcond=None)[0]
    offsets = {camera: float(value) for camera, value in zip(cameras, solution)}
    residuals = {pair: lag - (offsets[pair[0]] - offsets[pair[1]]) for pair, (lag, _) in summary.items()}
    return offsets, residuals
