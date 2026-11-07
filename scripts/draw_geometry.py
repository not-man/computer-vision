"""Наглядная проверка привязки: стены и метровая сетка на кадрах, вид камер сверху на плане."""
import argparse
from pathlib import Path

import cv2
import numpy as np

from mct.geometry import IMAGE_SIZE, Camera, load_cameras
from mct.plan import Plan


WALL_MARK_M = 1.5
DOOR_HEIGHT_M = 2.05


def visible_polyline(
    camera: Camera, start: np.ndarray, end: np.ndarray, steps: int = 60, height: float = 0.0, top: float | None = None,
    keep=None,
) -> list[np.ndarray]:
    """Участки отрезка, попадающие перед камерой и в пределы кадра, в координатах исходного кадра.

    height: высота над полом; top: высота второго конца, чтобы рисовать вертикальные рёбра;
    keep: условие на точку плана, например прямая видимость.
    """
    world = np.linspace(start, end, steps)
    heights = np.linspace(height, height if top is None else top, steps)
    h = camera.project_h(np.column_stack([world, heights]))
    pieces, current = [], []
    for point, (u, v, z) in zip(world, h):
        inside = False
        if z > 0.05 and (keep is None or keep(point)):
            distorted = camera.lens.distort(np.array([[u / z, v / z]]))[0]
            inside = -40 <= distorted[0] <= IMAGE_SIZE[0] + 40 and -40 <= distorted[1] <= IMAGE_SIZE[1] + 40
        if inside:
            current.append(distorted)
        elif current:
            pieces.append(np.array(current))
            current = []
    if current:
        pieces.append(np.array(current))
    return pieces


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("configs/plan_features.yaml"))
    parser.add_argument("--plan", type=Path, default=Path("data/office_camera.png"))
    parser.add_argument("--backgrounds", type=Path, default=Path("artifacts/geometry/background"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/geometry/overlay"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    plan_model = Plan.load(args.features)
    scale = plan_model.px_per_m
    cameras = load_cameras()
    plan = cv2.imread(str(args.plan))
    plan_height, plan_width = plan.shape[:2]

    for name, camera in cameras.items():
        image = cv2.imread(str(args.backgrounds / f"{name}.png"))
        viewer = np.array([camera.x, camera.y])

        # Рисуем только то, что камера может видеть по плану; сетку только на полу
        def on_floor(point: np.ndarray) -> bool:
            return plan_model.floor_margin(point) >= 0 and plan_model.line_of_sight(viewer, point)

        def seen(point: np.ndarray) -> bool:
            return plan_model.line_of_sight(viewer, point)

        for offset in np.arange(-25, 26):
            for start, end in (
                ((np.floor(camera.x) + offset, camera.y - 25), (np.floor(camera.x) + offset, camera.y + 25)),
                ((camera.x - 25, np.floor(camera.y) + offset), (camera.x + 25, np.floor(camera.y) + offset)),
            ):
                for piece in visible_polyline(camera, np.array(start), np.array(end), steps=400, keep=on_floor):
                    cv2.polylines(image, [piece.round().astype(np.int32)], False, (0, 200, 0), 1, cv2.LINE_AA)
        # Мебель стоит перед стенами, поэтому линия пола идёт сквозь неё. Вторая линия на высоте
        # показывает плоскость стены там, где низ закрыт
        for start, end in plan_model.wall_pieces():
            for piece in visible_polyline(camera, start, end, steps=120, keep=seen):
                cv2.polylines(image, [piece.round().astype(np.int32)], False, (0, 0, 255), 2, cv2.LINE_AA)
            for piece in visible_polyline(camera, start, end, steps=120, height=WALL_MARK_M, keep=seen):
                cv2.polylines(image, [piece.round().astype(np.int32)], False, (0, 140, 255), 1, cv2.LINE_AA)
        # Дверь рисуем целиком: её рамка на кадре сразу показывает, совпал ли проём
        for start, end in plan_model.doors:
            if not seen((start + end) / 2):
                continue
            frame = [(start, end, 0.0, None), (start, end, DOOR_HEIGHT_M, None), (start, start, 0.0, DOOR_HEIGHT_M), (end, end, 0.0, DOOR_HEIGHT_M)]
            for a, b, height, top in frame:
                for piece in visible_polyline(camera, a, b, steps=12, height=height, top=top):
                    cv2.polylines(image, [piece.round().astype(np.int32)], False, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.imwrite(str(args.out / f"{name}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 88])

        # Вид сверху: каждая точка плана берёт цвет из кадра
        xs, ys = np.meshgrid(np.arange(plan_width), np.arange(plan_height))
        world = np.stack([xs.ravel(), ys.ravel()], axis=1) / scale
        h = camera.project_h(np.column_stack([world, np.zeros(len(world))]))
        valid = h[:, 2] > 0.2
        rectified = np.full((len(world), 2), 1e6)
        rectified[valid] = h[valid, :2] / h[valid, 2:]
        source = camera.lens.distort(rectified).reshape(plan_height, plan_width, 2).astype(np.float32)
        far = np.hypot(world[:, 0] - camera.x, world[:, 1] - camera.y).reshape(plan_height, plan_width) > 14
        source[far | ~valid.reshape(plan_height, plan_width)] = -100
        warped = cv2.remap(cv2.imread(str(args.backgrounds / f"{name}.png")), source[..., 0], source[..., 1], cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        covered = warped.sum(axis=2) > 0
        top = plan.copy()
        top[covered] = (0.25 * plan[covered] + 0.75 * warped[covered]).astype(np.uint8)
        cv2.imwrite(str(args.out / f"top_{name}.jpg"), top, [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    main()
