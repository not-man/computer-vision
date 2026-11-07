import numpy as np

from mct.render import CANVAS, Stamps
from mct.showcase import BAR_HEIGHT, Moment, pick_moments, reel_seconds, with_bar


def track(track_id: int, start_s: float, end_s: float, cameras: tuple[str, ...] = ("camera_new_1",), yellow: bool = False,
          x_m: float = 10.0, hints: list[dict] | None = None) -> dict:
    return {
        "track_id": track_id, "t_start_ms": int(start_s * 1000), "t_end_ms": int(end_s * 1000), "cameras": list(cameras),
        "links": [{"confidence": "yellow" if yellow else "green"}],
        "segments": [{"kind": "observed", "camera_id": camera, "t_start_ms": int((start_s + index) * 1000), "t_end_ms": int(end_s * 1000)}
                     for index, camera in enumerate(cameras)],
        "plan_path": [[x_m, 10.0, int(start_s * 1000)], [x_m, 10.0, int(end_s * 1000)]],
        "possibly_same_person_as": hints or [],
    }


ALL_CAMERAS = tuple(f"camera_new_{index}" for index in (1, 2, 3, 5, 4))
THREE_CAMERAS = ALL_CAMERAS[:3]


def titles(moments: list[Moment]) -> list[str]:
    return [moment.title for moment in moments]


def test_walk_takes_the_widest_clean_track_and_shows_the_real_camera_order() -> None:
    tracks = [track(1, 0.0, 20.0, ALL_CAMERAS, yellow=True), track(2, 100.0, 120.0, ALL_CAMERAS), track(3, 200.0, 215.0, THREE_CAMERAS)]
    walk = pick_moments(tracks)[0]
    assert walk.spans == ((99.0, 121.0),)
    assert "1 → 2 → 3 → 5 → 4" in walk.note


def test_pair_needs_two_tracks_walking_close_together() -> None:
    close = [track(1, 0.0, 20.0, THREE_CAMERAS, x_m=10.0), track(2, 1.0, 21.0, THREE_CAMERAS, x_m=11.0)]
    apart = [track(1, 0.0, 20.0, THREE_CAMERAS, x_m=10.0), track(2, 1.0, 21.0, THREE_CAMERAS, x_m=30.0)]
    assert "Двое рядом: у каждого свой номер" in titles(pick_moments(close))
    assert "Двое рядом: у каждого свой номер" not in titles(pick_moments(apart))


def test_return_ignores_short_absence_and_takes_the_strongest_hint() -> None:
    tracks = [
        track(1, 0.0, 10.0), track(2, 20.0, 30.0), track(3, 300.0, 310.0, hints=[{"track_id": 1, "r": 0.7, "absent_s": 290.0}]),
        track(4, 400.0, 410.0, hints=[{"track_id": 2, "r": 0.9, "absent_s": 370.0}]),
        track(5, 415.0, 420.0, hints=[{"track_id": 4, "r": 0.99, "absent_s": 5.0}]),
    ]
    back = pick_moments(tracks)[-1]
    assert back.spans == ((26.0, 31.0), (399.0, 404.0))
    assert back.card == "прошло 6 мин 10 с"
    assert "#2" in back.note and "#4" in back.note


def test_crowd_does_not_repeat_frames_of_the_other_moments() -> None:
    tracks = [track(1, 0.0, 20.0, ALL_CAMERAS)] + [track(index, 5.0, 15.0) for index in range(2, 8)] + [track(8, 40.0, 60.0), track(9, 41.0, 59.0)]
    moments = pick_moments(tracks)
    crowd = next(moment for moment in moments if moment.title == "Много людей одновременно")
    assert crowd.spans[0][0] >= moments[0].spans[0][1]


def test_reel_stays_within_the_limit_in_the_worst_case() -> None:
    tracks = [
        track(1, 0.0, 24.0, ALL_CAMERAS), track(2, 100.0, 124.0, THREE_CAMERAS), track(3, 100.0, 124.0, THREE_CAMERAS),
        track(4, 500.0, 510.0), track(5, 900.0, 910.0, hints=[{"track_id": 4, "r": 0.9, "absent_s": 390.0}]),
    ]
    moments = pick_moments(tracks)
    assert len(moments) == 4
    assert reel_seconds(moments) <= 90.0


def test_bar_sits_above_the_untouched_frame_and_keeps_even_size() -> None:
    frame = np.full((CANVAS[1], CANVAS[0], 3), 7, np.uint8)
    result = with_bar(frame, Moment("Заголовок", "Пояснение", ((0.0, 1.0),)), 1, 4, Stamps())
    assert result.shape == (CANVAS[1] + BAR_HEIGHT, CANVAS[0], 3)
    assert result.shape[0] % 2 == 0
    assert np.array_equal(result[BAR_HEIGHT:], frame)


def test_short_return_is_shown_and_the_reel_still_fits() -> None:
    returned = track(6, 300.0, 340.0)
    returned["links"].append({"confidence": "green", "tier": "return", "t_ms": 310000, "contact_s": -20.0, "r": 0.9})
    tracks = [
        track(1, 0.0, 24.0, ALL_CAMERAS), track(2, 100.0, 124.0, THREE_CAMERAS), track(3, 100.0, 124.0, THREE_CAMERAS),
        track(4, 500.0, 510.0), track(5, 900.0, 910.0, hints=[{"track_id": 4, "r": 0.9, "absent_s": 390.0}]), returned,
    ]
    moments = pick_moments(tracks)
    assert "Вернулся быстрее минуты: номер тот же" in titles(moments)
    assert reel_seconds(moments) <= 90.0
