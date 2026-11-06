"""План этажа: масштаб, стены с дверными проёмами и граница пола. Координаты в метрах."""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from mct.trinity import Zone

FEATURES = Path(__file__).resolve().parents[2] / "configs" / "plan_features.yaml"
DOOR_POCKET_M = 1.5  # за дверью человек ещё считается настоящим: он в проёме или виден через него
DOOR_ZONE_RADIUS_M = 1.0
ON_WALL_TOLERANCE_M = 0.25


def _segments_cross(p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray) -> bool:
    def side(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    return side(p1, p2, q1) * side(p1, p2, q2) < 0 and side(q1, q2, p1) * side(q1, q2, p2) < 0


@dataclass(frozen=True)
class Plan:
    px_per_m: float
    walls: np.ndarray  # K×2×2
    doors: np.ndarray  # D×2×2, концы проёма
    walkable: np.ndarray  # N×2, многоугольник пола

    @classmethod
    def load(cls, path: Path = FEATURES) -> "Plan":
        features = yaml.safe_load(path.read_text())
        scale = features["plan_px_per_m"]
        return cls(
            scale,
            np.array(features["walls"], np.float64) / scale,
            np.array(features["doors"], np.float64) / scale,
            np.array(features["walkable"], np.float64) / scale,
        )

    def floor_margin(self, xy: np.ndarray) -> float:
        """Сколько метров от точки до границы пола: плюс внутри, минус снаружи. Карманы у дверей считаются полом."""
        contour = self.walkable.astype(np.float32).reshape(-1, 1, 2)
        inside = cv2.pointPolygonTest(contour, (float(xy[0]), float(xy[1])), True)
        centers = self.doors.mean(axis=1)
        pocket = DOOR_POCKET_M - np.hypot(centers[:, 0] - xy[0], centers[:, 1] - xy[1]).min()
        return float(max(inside, pocket))

    def door_zones(self) -> list[Zone]:
        centers = self.doors.mean(axis=1)
        return [Zone("DOOR", float(x), float(y), DOOR_ZONE_RADIUS_M) for x, y in centers]

    def line_of_sight(self, viewer: np.ndarray, target: np.ndarray, tolerance: float = 0.2) -> bool:
        """Видна ли точка пола с точки viewer: отрезок не пересекает стену, дверные проёмы открыты.

        Концы отрезка на tolerance метров не проверяются: камеры и сами стены стоят на линиях стен.
        """
        span = target - viewer
        length = float(np.linalg.norm(span))
        if length <= 2 * tolerance:
            return True
        direction = span / length
        start, end = viewer + direction * tolerance, target - direction * tolerance
        for a, b in self.wall_pieces():
            if _segments_cross(start, end, a, b):
                return False
        return True

    def wall_pieces(self) -> list[np.ndarray]:
        """Участки стен без дверных проёмов."""
        pieces = []
        for start, end in self.walls:
            length = float(np.linalg.norm(end - start))
            direction = (end - start) / length
            normal = np.array([-direction[1], direction[0]])
            cuts = []
            for door_a, door_b in self.doors:
                near_wall = max(abs((door_a - start) @ normal), abs((door_b - start) @ normal)) <= ON_WALL_TOLERANCE_M
                if near_wall:
                    low, high = sorted(((door_a - start) @ direction, (door_b - start) @ direction))
                    if high > 0 and low < length:
                        cuts.append((max(low, 0.0), min(high, length)))
            position = 0.0
            for low, high in sorted(cuts):
                if low > position:
                    pieces.append(np.array([start + direction * position, start + direction * low]))
                position = max(position, high)
            if position < length:
                pieces.append(np.array([start + direction * position, end]))
        return pieces
