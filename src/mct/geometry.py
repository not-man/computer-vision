"""Геометрия камер: снятие дисторсии объектива и физическая модель камеры над плоскостью пола.

Координаты мира: X и Y в метрах вдоль осей плана (x вправо, y вниз), height в метрах над полом.
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

IMAGE_SIZE = (1296, 972)  # координаты наблюдений из mct.perception
RADIUS = float(np.hypot(IMAGE_SIZE[0], IMAGE_SIZE[1]) / 2)
CONFIG = Path(__file__).resolve().parents[2] / "configs" / "cameras.yaml"


@dataclass(frozen=True)
class Lens:
    """Радиальная модель в форме деления: r_u = r_d / (1 + k1·r² + k2·r⁴), r в долях полудиагонали кадра."""

    k1: float
    k2: float = 0.0
    cx: float = IMAGE_SIZE[0] / 2
    cy: float = IMAGE_SIZE[1] / 2

    def _radial(self, r_d: np.ndarray) -> np.ndarray:
        return r_d / (1 + self.k1 * r_d**2 + self.k2 * r_d**4)

    def is_valid(self, r_max: float = 1.05) -> bool:
        """Модель обратима, только если r_u монотонно растёт на всём кадре."""
        r_d = np.linspace(0, r_max, 400)
        denominator = 1 + self.k1 * r_d**2 + self.k2 * r_d**4
        return bool((denominator > 0.05).all() and (np.diff(self._radial(r_d)) > 0).all())

    def undistort(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, np.float64).reshape(-1, 2)
        centered = points - (self.cx, self.cy)
        r2 = (centered**2).sum(axis=1, keepdims=True) / RADIUS**2
        return (self.cx, self.cy) + centered / (1 + self.k1 * r2 + self.k2 * r2**2)

    def distort(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, np.float64).reshape(-1, 2)
        centered = points - (self.cx, self.cy)
        r_u = np.hypot(centered[:, 0], centered[:, 1]) / RADIUS
        table_d = np.linspace(0, 1.05, 4000)
        r_d = np.interp(r_u, self._radial(table_d), table_d)
        ratio = np.divide(r_d, r_u, out=np.ones_like(r_u), where=r_u > 1e-9)
        return (self.cx, self.cy) + centered * ratio[:, None]

    def rectify(self, image: np.ndarray, zoom: float, size: tuple[int, int]) -> np.ndarray:
        """Кадр без дисторсии. Точка кадра u попадает в ((u - c)·zoom + size/2)."""
        width, height = size
        xs, ys = np.meshgrid(np.arange(width), np.arange(height))
        view = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)
        undistorted = (view - (width / 2, height / 2)) / zoom + (self.cx, self.cy)
        source = self.distort(undistorted).reshape(height, width, 2).astype(np.float32)
        return cv2.remap(image, source[..., 0], source[..., 1], cv2.INTER_LINEAR, borderValue=(40, 40, 40))


@dataclass(frozen=True)
class Camera:
    """Камера-обскура после снятия дисторсии. yaw: куда смотрит на плане, tilt: наклон вниз от горизонта."""

    lens: Lens
    f: float
    x: float
    y: float
    height: float
    yaw: float
    tilt: float
    roll: float

    def rotation(self) -> np.ndarray:
        """Строки: оси камеры (вправо, вниз, вперёд) в осях (X, Y, вниз)."""
        heading = np.array([np.cos(self.yaw), np.sin(self.yaw), 0.0])
        forward = np.cos(self.tilt) * heading + np.array([0.0, 0.0, np.sin(self.tilt)])
        right = np.array([-np.sin(self.yaw), np.cos(self.yaw), 0.0])
        down = np.cross(forward, right)
        roll_cos, roll_sin = np.cos(self.roll), np.sin(self.roll)
        return np.stack([roll_cos * right + roll_sin * down, -roll_sin * right + roll_cos * down, forward])

    def camera_matrix(self) -> np.ndarray:
        return np.array([[self.f, 0, self.lens.cx], [0, self.f, self.lens.cy], [0, 0, 1.0]])

    def project_h(self, world: np.ndarray) -> np.ndarray:
        """Однородные координаты кадра без дисторсии; точка (X, Y, height). z <= 0 означает «за камерой»."""
        world = np.asarray(world, np.float64).reshape(-1, 3)
        relative = np.stack([world[:, 0] - self.x, world[:, 1] - self.y, self.height - world[:, 2]], axis=1)
        return (self.camera_matrix() @ self.rotation() @ relative.T).T

    def project(self, world: np.ndarray) -> np.ndarray:
        h = self.project_h(world)
        return h[:, :2] / h[:, 2:]

    def floor_homography(self) -> np.ndarray:
        """Матрица из (X, Y, 1) на полу в однородные координаты кадра без дисторсии."""
        rotation = self.rotation()
        translation = -rotation @ np.array([self.x, self.y, -self.height])
        return self.camera_matrix() @ np.stack([rotation[:, 0], rotation[:, 1], translation], axis=1)

    def rectified_to_floor(self, points: np.ndarray) -> np.ndarray:
        """Точки кадра без дисторсии в (X, Y) на полу. Выше горизонта результат не имеет смысла."""
        points = np.asarray(points, np.float64).reshape(-1, 2)
        h = np.linalg.solve(self.floor_homography(), np.column_stack([points, np.ones(len(points))]).T).T
        return h[:, :2] / h[:, 2:]

    def image_to_floor(self, points: np.ndarray) -> np.ndarray:
        """Точки исходного кадра (1296×972) в (X, Y) на полу."""
        return self.rectified_to_floor(self.lens.undistort(points))

    def floor_to_image(self, points: np.ndarray) -> np.ndarray:
        """Точки пола (X, Y) в исходный кадр (1296×972)."""
        points = np.asarray(points, np.float64).reshape(-1, 2)
        world = np.column_stack([points, np.zeros(len(points))])
        return self.lens.distort(self.project(world))

    def nadir(self) -> np.ndarray:
        """Точка схода вертикалей в кадре без дисторсии: туда смотрят ноги всех стоящих людей."""
        direction = self.camera_matrix() @ self.rotation()[:, 2]
        return direction[:2] / direction[2]

    def foot_and_head(self, mask_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Ноги и макушка по маске, в кадре без дисторсии: крайние точки вдоль оси от точки схода вертикалей.

        У наклонной камеры человек вытянут к точке схода, поэтому низ рамки на краю кадра ногами не является.
        """
        points = self.lens.undistort(mask_xy)
        nadir = self.nadir()
        axis = points.mean(axis=0) - nadir
        axis /= np.linalg.norm(axis)
        along = (points - nadir) @ axis
        return nadir + axis * np.percentile(along, 1), nadir + axis * np.percentile(along, 99)

    def floor_from_head(self, head: np.ndarray, person_height: float = 1.72) -> np.ndarray:
        """Точка пола под макушкой в кадре без дисторсии: для людей, у которых ноги закрыты или обрезаны."""
        ray = np.linalg.solve(self.camera_matrix() @ self.rotation(), np.array([head[0], head[1], 1.0]))
        # ray[2] > 0: луч идёт вниз; иначе макушка выше уровня камеры и точку определить нельзя
        distance = (self.height - person_height) / max(ray[2], 1e-9)
        return np.array([self.x + ray[0] * distance, self.y + ray[1] * distance])

    def heights_of(self, foot: np.ndarray, head: np.ndarray) -> np.ndarray:
        """Высоты над полом точек head, стоящих строго над точками foot; всё в кадре без дисторсии."""
        foot, head = np.asarray(foot, np.float64).reshape(-1, 2), np.asarray(head, np.float64).reshape(-1, 2)
        horizontal = self.rectified_to_floor(foot) - (self.x, self.y)
        rays = np.linalg.solve(self.camera_matrix() @ self.rotation(), np.column_stack([head, np.ones(len(head))]).T).T
        distance = np.hypot(horizontal[:, 0], horizontal[:, 1])
        along = (rays[:, :2] * horizontal).sum(axis=1) / np.maximum(distance, 1e-9)
        # Луч (along, down): падение луча на горизонтальном пути distance даёт высоту точки над полом
        return self.height - distance * rays[:, 2] / np.maximum(along, 1e-9)

    def height_of(self, foot: np.ndarray, head: np.ndarray) -> float:
        return float(self.heights_of(foot, head)[0])


def edge_chains(image: np.ndarray, min_extent: int = 90, top_margin: int = 40) -> list[np.ndarray]:
    """Цепочки краёв без развилок: развилки вырезаются, чтобы цепочка была одной линией."""
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    edges = (cv2.Canny(gray, 40, 120) > 0).astype(np.uint8)
    edges[:top_margin] = 0  # наложенные часы
    neighbours = cv2.filter2D(edges, -1, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT) - edges
    edges[(edges > 0) & (neighbours > 2)] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
    chains = []
    for label in range(1, count):
        if max(stats[label, cv2.CC_STAT_WIDTH], stats[label, cv2.CC_STAT_HEIGHT]) >= min_extent:
            ys, xs = np.nonzero(labels == label)
            chains.append(np.stack([xs, ys], axis=1).astype(np.float64))
    return chains


def straightness(points: np.ndarray) -> tuple[float, float]:
    """Отклонение от прямой, делённое на длину, и сама длина."""
    centered = points - points.mean(axis=0)
    eigenvalues = np.linalg.eigvalsh(centered.T @ centered / len(points))
    deviation = np.sqrt(max(eigenvalues[0], 0.0))
    length = np.sqrt(12 * max(eigenvalues[1], 1e-9))
    return float(deviation / length), float(length)


def load_cameras(path: Path = CONFIG) -> dict[str, Camera]:
    config = yaml.safe_load(path.read_text())["cameras"]
    cameras = {}
    for name, values in config.items():
        cameras[name] = Camera(lens=Lens(**values["lens"]), **{k: v for k, v in values.items() if k != "lens"})
    return cameras
