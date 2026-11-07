import numpy as np
import pandas as pd

from mct.tracklets import FilterConfig, _set_embeddings, build_tracklets, find_identity_jumps, find_tail_swaps
from mct.trinity import Tracklet


def observations(camera: str, raw_track: int, t: np.ndarray, x: np.ndarray, **overrides) -> pd.DataFrame:
    table = pd.DataFrame({
        "camera": camera, "raw_track": raw_track, "frame": np.round(t * 25).astype(int), "t_s": t, "X": x, "Y": 5.0,
        "cov_xx": 0.01, "cov_xy": 0.0, "cov_yy": 0.01, "conf": 0.9, "person_height": 1.7, "margin": 1.0,
        "touches_edge": False, "source": "foot",
        # Рамка едет вместе с человеком, поэтому неподвижность видна и по кадру
        "x1": 100 + 20 * x, "y1": 100.0, "x2": 140 + 20 * x, "y2": 220.0,
    })
    return table.assign(**overrides)


T = np.arange(0.0, 6.0, 0.1)


def test_walker_passes_and_reflection_behind_wall_is_dropped() -> None:
    walker = observations("a", 1, T, 1.0 + 1.2 * T)
    mirror = observations("a", 2, T, 1.0 + 1.2 * T, margin=-1.5, conf=0.9)
    tracklets, table, dropped = build_tracklets(pd.concat([walker, mirror]), {})
    assert [x.camera for x in tracklets] == ["a"]
    assert dropped.reason.str.contains("вне пола").all() and len(dropped) == 1
    assert set(table.raw_track) == {1}


def test_low_still_object_is_furniture_but_low_walker_is_a_person() -> None:
    chair = observations("a", 1, T, np.full(len(T), 3.0), person_height=1.1)
    behind_desk = observations("a", 2, T, 1.0 + 1.2 * T, person_height=1.1)
    tracklets, _, dropped = build_tracklets(pd.concat([chair, behind_desk]), {})
    assert len(tracklets) == 1
    assert "мебель" in dropped.reason.iloc[0]


def test_cut_off_person_at_frame_edge_is_not_judged_by_height() -> None:
    cut = observations("a", 1, T, np.full(len(T), 3.0), person_height=0.8, touches_edge=True)
    assert len(build_tracklets(cut, {})[0]) == 1


def test_weak_detection_is_dropped() -> None:
    assert build_tracklets(observations("a", 1, T, 1.0 + 1.2 * T, conf=0.2), {})[0] == []


def test_identity_switch_cuts_the_track_but_a_short_foot_jitter_does_not() -> None:
    x = 1.0 + 1.2 * T
    switched = x.copy()
    switched[30:] += 8.0
    assert len(build_tracklets(observations("a", 1, T, switched), {})[0]) == 2
    jitter = x.copy()
    jitter[30:32] += 3.0
    assert len(build_tracklets(observations("a", 1, T, jitter), {})[0]) == 1


def test_unreliable_positions_do_not_enter_the_path_but_stay_labelled() -> None:
    table = observations("a", 1, T, 1.0 + 1.2 * T)
    table.loc[table.index[:5], "source"] = "unreliable"
    table.loc[table.index[:5], "X"] = 50.0
    tracklets, assigned, _ = build_tracklets(table, {})
    assert len(tracklets) == 1 and tracklets[0].xy[:, 0].max() < 10
    assert (assigned.source == "unreliable").sum() == 5


def test_filter_thresholds_are_configurable() -> None:
    mirror = observations("a", 2, T, 1.0 + 1.2 * T, margin=-1.5)
    assert len(build_tracklets(mirror, {}, FilterConfig(out_of_floor_m=-2.0))[0]) == 1


def tracklet(id_: int, camera: str) -> Tracklet:
    return Tracklet(id_, camera, np.array([0.0, 1.0]), np.zeros((2, 2)), np.tile(np.eye(2), (2, 1, 1)))


def test_camera_tint_is_removed_from_appearance() -> None:
    # У камер сильный общий оттенок: без вычитания разные люди на них похожи, после вычитания нет
    tint_a, tint_b = np.array([3.0, 0.0, 0.0, 0.0]), np.array([0.0, 3.0, 0.0, 0.0])
    person_1, person_2 = np.array([0.0, 0.0, 1.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0])
    items = [tracklet(i, camera) for i, camera in ((1, "a"), (2, "a"), (3, "b"), (4, "b"))]
    appearance = {1: (tint_a + person_1)[None], 2: (tint_a + person_2)[None], 3: (tint_b + person_1)[None], 4: (tint_b + person_2)[None]}
    _set_embeddings(items, appearance)
    a1, a2, b1, b2 = (item.embedding for item in items)
    assert a1 @ b1 > 0.9
    assert a1 @ b2 < -0.9
    assert a1 @ a2 < -0.9


SHORT = np.arange(0.0, 2.0, 0.1)


def standing(raw_track: int, start_s: float, **overrides) -> pd.DataFrame:
    values = {"conf": 0.5, **overrides}
    return observations("a", raw_track, SHORT + start_s, np.full(len(SHORT), 3.0), **values)


def test_object_that_detector_keeps_returning_to_is_dropped() -> None:
    table = pd.concat([standing(i, 100.0 * i) for i in range(1, 5)])
    tracklets, _, dropped = build_tracklets(table, {})
    assert tracklets == []
    assert dropped.reason.str.contains("предмет").all() and len(dropped) == 4


def test_standing_once_or_twice_is_not_an_object() -> None:
    assert len(build_tracklets(pd.concat([standing(1, 100.0), standing(2, 200.0)]), {})[0]) == 2


def test_confident_person_returning_to_the_same_spot_is_kept() -> None:
    table = pd.concat([standing(i, 100.0 * i, conf=0.9) for i in range(1, 5)])
    assert len(build_tracklets(table, {})[0]) == 4


def test_people_walking_through_the_same_place_are_kept() -> None:
    table = pd.concat([observations("a", i, T + 100.0 * i, 1.0 + 1.2 * T, conf=0.5) for i in range(1, 5)])
    assert len(build_tracklets(table, {})[0]) == 4


def test_floor_tolerance_grows_with_position_error() -> None:
    near = observations("a", 1, T, 1.0 + 1.2 * T, margin=-1.5)
    far = observations("a", 1, T, 1.0 + 1.2 * T, margin=-1.5, cov_xx=0.5, cov_yy=0.5)
    assert build_tracklets(near, {})[0] == []
    assert len(build_tracklets(far, {})[0]) == 1


def test_weak_detection_needs_only_one_repeat() -> None:
    table = pd.concat([standing(1, 100.0, conf=0.3), standing(2, 200.0, conf=0.3, conf_peak=0.0)])
    tracklets, _, dropped = build_tracklets(table.drop(columns="conf_peak", errors="ignore").assign(conf=lambda t: t.conf.clip(lower=0.36)), {})
    assert tracklets == []
    assert dropped.reason.str.contains("предмет").all()


def test_person_with_hidden_legs_just_past_the_floor_edge_is_kept() -> None:
    near = observations("a", 1, T, 1.0 + 1.2 * T, margin=-0.05, person_height=1.1)
    mirror = observations("a", 1, T, 1.0 + 1.2 * T, margin=-0.24, person_height=1.1)
    assert len(build_tracklets(near, {})[0]) == 1
    assert build_tracklets(mirror, {})[0] == []


def test_reflection_inside_marked_mirror_is_dropped_but_person_passing_by_is_not() -> None:
    cfg = FilterConfig(mirrors=(("a", (90.0, 90.0, 300.0, 230.0)),))
    reflection = observations("a", 1, T, np.full(len(T), 3.0), x1=100.0, x2=140.0)
    passing = observations("a", 2, T, 1.0 + 1.2 * T, x1=100.0, y1=100.0, x2=140.0, y2=400.0)
    other_camera = observations("b", 3, T, np.full(len(T), 3.0), x1=100.0, x2=140.0)
    tracklets, _, dropped = build_tracklets(pd.concat([reflection, passing, other_camera]), {}, cfg)
    assert sorted(x.camera for x in tracklets) == ["a", "b"]
    assert "зеркало" in dropped.reason.iloc[0]


PERSON_1, PERSON_2 = np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0, 0.0])


def walking(id_: int, camera: str, start: float, end: float, embedding: np.ndarray | None, y: float = 0.0) -> Tracklet:
    t = np.arange(start, end + 1e-9, 0.2)
    return Tracklet(id_, camera, t, np.column_stack([t, np.full(len(t), y)]), np.tile(np.eye(2) * 0.01, (len(t), 1, 1)), embedding)


def jump_seconds(x: Tracklet) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    # До 5 с на куске один человек, после 5 с другой
    times = np.arange(x.start, x.end, 1.0)
    vectors = np.array([PERSON_1 if t < 5 else PERSON_2 for t in times])
    return {x.id: ((times * 10).astype(int), times, vectors)}


def test_jump_is_found_when_each_half_has_its_own_person_nearby() -> None:
    x = walking(1, "a", 0.0, 10.0, (PERSON_1 + PERSON_2) / np.sqrt(2))
    first, second = walking(2, "b", 0.0, 10.0, PERSON_1), walking(3, "b", 0.0, 10.0, PERSON_2, y=0.5)
    assert find_identity_jumps([x, first, second], jump_seconds(x), FilterConfig()) == {1: 5}


def test_change_of_look_without_other_people_is_not_a_jump() -> None:
    # Человек повернулся, и внешность изменилась: рядом нет двух разных людей, значит, резать нельзя
    x = walking(1, "a", 0.0, 10.0, (PERSON_1 + PERSON_2) / np.sqrt(2))
    assert find_identity_jumps([x], jump_seconds(x), FilterConfig()) == {}


def test_two_views_of_one_person_are_not_two_people() -> None:
    # Другие камеры видят того же человека спереди и сзади: они рядом друг с другом, поэтому доказательства нет
    x = walking(1, "a", 0.0, 10.0, (PERSON_1 + PERSON_2) / np.sqrt(2))
    front, back = walking(2, "b", 0.0, 10.0, PERSON_1), walking(3, "c", 0.0, 10.0, PERSON_2, y=0.2)
    assert find_identity_jumps([x, front, back], jump_seconds(x), FilterConfig()) == {}


def test_jump_needs_clearly_different_halves() -> None:
    x = walking(1, "a", 0.0, 10.0, PERSON_1)
    times = np.arange(0.0, 10.0, 1.0)
    same = {1: ((times * 10).astype(int), times, np.tile(PERSON_1, (len(times), 1)))}
    assert find_identity_jumps([x, walking(2, "b", 0.0, 10.0, PERSON_1), walking(3, "b", 0.0, 10.0, PERSON_2, y=0.5)], same, FilterConfig()) == {}


def with_boxes(x: Tracklet, box: tuple[float, float, float, float]) -> Tracklet:
    x.boxes = np.tile(box, (len(x.t), 1))
    return x


def swap_seconds(*ids: int) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    times = np.arange(0.0, 10.0, 1.0)
    one = np.array([PERSON_1 if t < 5 else PERSON_2 for t in times])
    other = np.array([PERSON_2 if t < 5 else PERSON_1 for t in times])
    return {ids[0]: ((times * 10).astype(int), times, one), ids[1]: ((times * 10).astype(int), times, other)}


def test_two_people_exchanging_numbers_are_cut_at_the_same_moment() -> None:
    a = with_boxes(walking(1, "a", 0.0, 10.0, PERSON_1), (0.0, 0.0, 40.0, 90.0))
    b = with_boxes(walking(2, "a", 0.0, 10.0, PERSON_2), (30.0, 0.0, 70.0, 90.0))
    assert find_tail_swaps([a, b], swap_seconds(1, 2), FilterConfig()) == {1: 5, 2: 5}


def test_exchange_needs_touching_boxes() -> None:
    a = with_boxes(walking(1, "a", 0.0, 10.0, PERSON_1), (0.0, 0.0, 40.0, 90.0))
    b = with_boxes(walking(2, "a", 0.0, 10.0, PERSON_2), (300.0, 0.0, 340.0, 90.0))
    assert find_tail_swaps([a, b], swap_seconds(1, 2), FilterConfig()) == {}


def test_exchange_needs_both_sides_to_change() -> None:
    a = with_boxes(walking(1, "a", 0.0, 10.0, PERSON_1), (0.0, 0.0, 40.0, 90.0))
    b = with_boxes(walking(2, "a", 0.0, 10.0, PERSON_2), (30.0, 0.0, 70.0, 90.0))
    times = np.arange(0.0, 10.0, 1.0)
    only_one_changes = {
        1: ((times * 10).astype(int), times, np.array([PERSON_1 if t < 5 else PERSON_2 for t in times])),
        2: ((times * 10).astype(int), times, np.tile(PERSON_2, (len(times), 1))),
    }
    assert find_tail_swaps([a, b], only_one_changes, FilterConfig()) == {}


def test_exchange_needs_high_cross_similarity() -> None:
    # Половины разные, а перекрёстное сходство только 0.45: так выглядит общий переход в другой свет
    a = with_boxes(walking(1, "a", 0.0, 10.0, PERSON_1), (0.0, 0.0, 40.0, 90.0))
    b = with_boxes(walking(2, "a", 0.0, 10.0, PERSON_2), (30.0, 0.0, 70.0, 90.0))
    rest = float(np.sqrt(1 - 0.2**2 - 0.45**2))
    tail_a, tail_b = np.array([-0.2, 0.45, rest, 0.0]), np.array([0.45, -0.2, 0.0, rest])
    times = np.arange(0.0, 10.0, 1.0)
    weak = {
        1: ((times * 10).astype(int), times, np.array([PERSON_1 if t < 5 else tail_a for t in times])),
        2: ((times * 10).astype(int), times, np.array([PERSON_2 if t < 5 else tail_b for t in times])),
    }
    assert find_tail_swaps([a, b], weak, FilterConfig()) == {}
    assert set(find_tail_swaps([a, b], weak, FilterConfig(swap_min_match=0.3))) == {1, 2}
