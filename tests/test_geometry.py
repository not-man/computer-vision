import numpy as np
import pytest

from mct.geometry import Camera, Lens
from mct.plan import Plan

LENS = Lens(k1=-0.52, k2=-0.05, cx=650.0, cy=487.0)
CAMERA = Camera(LENS, f=790.0, x=10.0, y=8.0, height=3.0, yaw=np.radians(60), tilt=np.radians(40), roll=np.radians(-5))


def test_lens_roundtrip() -> None:
    points = np.array([[650.0, 487.0], [100.0, 100.0], [1200.0, 900.0], [30.0, 815.0]])
    assert np.allclose(LENS.distort(LENS.undistort(points)), points, atol=0.05)


def test_floor_roundtrip() -> None:
    floor = np.array([[11.0, 12.0], [13.0, 11.0], [12.5, 13.5]])
    assert np.allclose(CAMERA.image_to_floor(CAMERA.floor_to_image(floor)), floor, atol=1e-2)


def test_vertical_lines_pass_through_nadir() -> None:
    foot, head = CAMERA.project(np.array([[12.0, 12.0, 0.0], [12.0, 12.0, 1.7]]))
    nadir = CAMERA.nadir()
    cross = (foot - nadir)[0] * (head - nadir)[1] - (foot - nadir)[1] * (head - nadir)[0]
    assert abs(cross) < 1e-6 * np.linalg.norm(foot - nadir) * np.linalg.norm(head - nadir)
    # Ноги ближе к точке схода, чем макушка
    assert np.linalg.norm(foot - nadir) < np.linalg.norm(head - nadir)


def test_person_height_is_recovered() -> None:
    foot, head = CAMERA.project(np.array([[12.0, 13.0, 0.0], [12.0, 13.0, 1.7]]))
    assert abs(CAMERA.height_of(foot, head) - 1.7) < 1e-6


def test_foot_and_head_from_tilted_silhouette() -> None:
    # Силуэт: точки вдоль вертикали над одной точкой пола, в исходном кадре с дисторсией
    heights = np.linspace(0.0, 1.7, 500)
    world = np.column_stack([np.full(500, 13.0), np.full(500, 11.0), heights])
    silhouette = LENS.distort(CAMERA.project(world))
    foot, head = CAMERA.foot_and_head(silhouette)
    assert np.allclose(CAMERA.rectified_to_floor(foot)[0], (13.0, 11.0), atol=0.1)
    assert abs(CAMERA.height_of(foot, head) - 1.7) < 0.1


def test_floor_from_head_recovers_position() -> None:
    head = CAMERA.project(np.array([[12.0, 13.0, 1.72]]))[0]
    assert np.allclose(CAMERA.floor_from_head(head), (12.0, 13.0), atol=1e-6)
    # Ошибка роста на 10 см сдвигает точку вдоль луча, но остаётся небольшой
    assert np.hypot(*(CAMERA.floor_from_head(head, 1.62) - (12.0, 13.0))) < 0.5


def test_floor_margin_and_door_pockets() -> None:
    plan = Plan(
        1.0,
        np.array([[[0.0, 0.0], [10.0, 0.0]]]),
        np.array([[[4.0, 0.0], [6.0, 0.0]]]),
        np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0]]),
    )
    assert plan.floor_margin(np.array([5.0, 2.5])) == pytest.approx(2.5)
    # Стена без двери: отражение за ней отсекается
    assert plan.floor_margin(np.array([1.0, -2.0])) == pytest.approx(-2.0)
    # Тот же отступ, но за проёмом: человек виден через дверь
    assert plan.floor_margin(np.array([5.0, -1.0])) > 0
    pieces = plan.wall_pieces()
    assert [(float(p[0][0]), float(p[1][0])) for p in pieces] == [(0.0, 4.0), (6.0, 10.0)]
    assert plan.door_zones()[0].contains(np.array([5.0, 0.5]))


def test_line_of_sight_is_blocked_by_wall_but_not_by_door() -> None:
    plan = Plan(
        1.0,
        np.array([[[0.0, 0.0], [10.0, 0.0]]]),
        np.array([[[4.0, 0.0], [6.0, 0.0]]]),
        np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0]]),
    )
    assert not plan.line_of_sight(np.array([1.0, 2.0]), np.array([1.0, -2.0]))
    assert plan.line_of_sight(np.array([5.0, 2.0]), np.array([5.0, -2.0]))
    assert plan.line_of_sight(np.array([1.0, 2.0]), np.array([9.0, 2.0]))
    # Сама точка на стене видна со своей стороны
    assert plan.line_of_sight(np.array([1.0, 2.0]), np.array([2.0, 0.0]))
