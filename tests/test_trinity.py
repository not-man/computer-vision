"""Сценарии раздела 6 DESIGN.md: что система обязана сделать, независимо от настройки на видео."""
import numpy as np
import pytest

from mct.trinity import LinkResult, Tracklet, TrinityConfig, Zone, ZoneMap, check_invariants, link_tracklets

CONFIG = TrinityConfig()


def look(angle_deg: float) -> np.ndarray:
    """Вектор внешности: сходство двух векторов равно косинусу разности углов."""
    return np.array([np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))])


SAME = look(10.0)  # сходство с look(0) 0.98
CLEARLY_OTHER = look(100.0)  # сходство с look(0) -0.17


def walker(
    tracklet_id: int,
    camera: str,
    t0: float,
    t1: float,
    start: tuple[float, float],
    velocity: tuple[float, float] = (1.4, 0.0),
    embedding: np.ndarray | None = None,
    sigma: float = 0.15,
    starts_at_edge: bool = False,
    ends_at_edge: bool = False,
) -> Tracklet:
    t = np.arange(t0, t1 + 1e-9, 0.1)
    noise = np.random.default_rng(tracklet_id).normal(0, 0.03, (len(t), 2))
    xy = np.array(start) + np.outer(t - t0, velocity) + noise
    cov = np.tile(np.eye(2) * sigma**2, (len(t), 1, 1))
    return Tracklet(tracklet_id, camera, t, xy, cov, embedding, starts_at_edge, ends_at_edge)


def groups(result: LinkResult) -> set[frozenset[int]]:
    return {frozenset(track.tracklets) for track in result.tracks}


def run(tracklets: list[Tracklet], zones: ZoneMap | None = None) -> LinkResult:
    result = link_tracklets(tracklets, CONFIG, zones or ZoneMap())
    assert check_invariants(result, tracklets, CONFIG) == []
    return result


# Обязательные сценарии из таблицы раздела 6


def test_fast_pass_with_100ms_contact_is_one_track() -> None:
    a = walker(1, "cam_1", 0.0, 10.0, (0.0, 5.0), ends_at_edge=True)
    b = walker(2, "cam_2", 9.9, 20.0, (13.86, 5.0), starts_at_edge=True)
    result = run([a, b])
    assert groups(result) == {frozenset({1, 2})}
    assert result.tracks[0].links[0].tier == "brief"


def test_fast_pass_with_300ms_gap_is_one_track() -> None:
    a = walker(1, "cam_1", 0.0, 10.0, (0.0, 5.0), ends_at_edge=True)
    b = walker(2, "cam_2", 10.3, 20.0, (14.42, 5.0), starts_at_edge=True)
    result = run([a, b])
    assert groups(result) == {frozenset({1, 2})}
    assert result.tracks[0].links[0].tier == "brief"


def test_walkers_passing_each_other_at_the_boundary_stay_apart() -> None:
    a = walker(1, "cam_1", 0.0, 10.0, (0.0, 5.0), ends_at_edge=True)
    b = walker(2, "cam_2", 10.0, 20.0, (14.0, 5.0), velocity=(-1.4, 0.0), starts_at_edge=True)
    assert groups(run([a, b])) == {frozenset({1}), frozenset({2})}


def test_one_leaves_while_other_enters_nearby_stay_apart() -> None:
    a = walker(1, "cam_1", 0.0, 10.0, (0.0, 5.0), embedding=look(0), ends_at_edge=True)
    b = walker(2, "cam_2", 10.2, 20.0, (14.3, 5.0), embedding=CLEARLY_OTHER, starts_at_edge=True)
    assert groups(run([a, b])) == {frozenset({1}), frozenset({2})}


def test_two_people_long_visible_and_briefly_close_stay_apart() -> None:
    # Пути пересекаются на секунду, остальное время люди далеко друг от друга
    a = walker(1, "cam_1", 0.0, 60.0, (0.0, 5.0))
    b = walker(2, "cam_2", 0.0, 60.0, (42.0, -37.0), velocity=(0.0, 1.4))
    assert groups(run([a, b])) == {frozenset({1}), frozenset({2})}


def test_two_people_side_by_side_on_one_camera_never_merge() -> None:
    a = walker(1, "cam_1", 0.0, 30.0, (0.0, 5.0), embedding=look(0))
    b = walker(2, "cam_1", 0.0, 30.0, (0.0, 5.3), embedding=look(0))
    assert groups(run([a, b])) == {frozenset({1}), frozenset({2})}


def test_indistinguishable_side_by_side_handoff_is_yellow_with_correct_count() -> None:
    a1 = walker(1, "cam_1", 0.0, 10.0, (0.0, 5.0), embedding=look(0))
    a2 = walker(2, "cam_1", 0.0, 10.0, (0.0, 5.4), embedding=look(0))
    b1 = walker(3, "cam_2", 8.0, 20.0, (11.2, 5.0), embedding=look(0))
    b2 = walker(4, "cam_2", 8.0, 20.0, (11.2, 5.4), embedding=look(0))
    result = run([a1, a2, b1, b2])
    assert len(result.tracks) == 2
    assert all({"cam_1", "cam_2"} == set(track.cameras) for track in result.tracks)
    assert all(link.confidence == "yellow" for track in result.tracks for link in track.links)


def test_person_behind_column_for_3s_is_one_track() -> None:
    a = walker(1, "cam_1", 0.0, 10.0, (3.0, 5.0), velocity=(0.7, 0.0), embedding=look(0))
    b = walker(2, "cam_1", 13.0, 20.0, (12.0, 5.0), velocity=(0.7, 0.0), embedding=SAME)
    zones = ZoneMap([Zone("OCCLUDER", 11.0, 5.0, 1.2)])
    result = run([a, b], zones)
    assert groups(result) == {frozenset({1, 2})}
    assert result.tracks[0].links[0].tier == "gap"
    assert [s.kind for s in result.tracks[0].segments] == ["observed", "occluded", "observed"]


def test_left_through_door_and_back_after_a_minute_is_new_id_with_hint() -> None:
    a = walker(1, "cam_1", 0.0, 10.0, (0.0, 5.0), embedding=look(0))
    b = walker(2, "cam_1", 70.0, 80.0, (14.0, 5.0), velocity=(-1.4, 0.0), embedding=SAME)
    zones = ZoneMap([Zone("DOOR", 14.0, 5.0, 1.0)])
    result = run([a, b], zones)
    assert groups(result) == {frozenset({1}), frozenset({2})}
    first, second = result.tracks
    assert first.close_zone == "DOOR"
    assert second.hints[0].track_id == first.track_id
    assert second.hints[0].absent_s == pytest.approx(60.0)


# Правила буфера


def test_missing_column_marking_splits_after_open_buffer() -> None:
    # Единственный пропуск зоны с заметным вредом: без разметки разрыв 3 с длиннее буфера OPEN
    a = walker(1, "cam_1", 0.0, 10.0, (3.0, 5.0), velocity=(0.7, 0.0), embedding=look(0))
    b = walker(2, "cam_1", 13.0, 20.0, (12.0, 5.0), velocity=(0.7, 0.0), embedding=SAME)
    assert groups(run([a, b])) == {frozenset({1}), frozenset({2})}


def test_short_gap_in_open_area_is_one_track_without_reid() -> None:
    a = walker(1, "cam_1", 0.0, 10.0, (3.0, 5.0))
    b = walker(2, "cam_1", 11.0, 20.0, (18.4, 5.0))
    assert groups(run([a, b])) == {frozenset({1, 2})}


def test_door_within_buffer_needs_matching_appearance() -> None:
    a = walker(1, "cam_1", 0.0, 10.0, (0.0, 5.0), embedding=look(0))
    same = walker(2, "cam_1", 11.0, 20.0, (14.5, 5.0), embedding=SAME)
    other = walker(2, "cam_1", 11.0, 20.0, (14.5, 5.0), embedding=CLEARLY_OTHER)
    no_look = walker(2, "cam_1", 11.0, 20.0, (14.5, 5.0))
    zones = ZoneMap([Zone("DOOR", 14.0, 5.0, 1.0)])
    assert groups(run([a, same], zones)) == {frozenset({1, 2})}
    assert groups(run([a, other], zones)) == {frozenset({1}), frozenset({2})}
    assert groups(run([a, no_look], zones)) == {frozenset({1}), frozenset({2})}


def test_far_fragment_with_huge_error_cannot_jump_by_many_meters() -> None:
    # Обломок вдали: погрешность 4 м не должна разрешать скачок на 8 м за секунду
    a = walker(1, "cam_1", 0.0, 10.0, (0.0, 5.0), sigma=4.0, embedding=look(0))
    b = walker(2, "cam_2", 11.0, 20.0, (22.0, 5.0), sigma=4.0, embedding=SAME)
    assert groups(run([a, b])) == {frozenset({1}), frozenset({2})}


def test_brief_contact_with_huge_error_needs_small_distance() -> None:
    a = walker(1, "cam_1", 0.0, 10.0, (0.0, 5.0), sigma=4.0, ends_at_edge=True)
    b = walker(2, "cam_2", 9.9, 20.0, (18.0, 5.0), sigma=4.0, starts_at_edge=True)
    assert groups(run([a, b])) == {frozenset({1}), frozenset({2})}


def test_almost_adjacent_at_door_needs_only_no_conflict_in_appearance() -> None:
    # Человек стоит у двери, пропал на 1.2 с и появился там же. Снимков мало, поэтому сходство низкое
    a = walker(1, "cam_1", 0.0, 10.0, (14.0, 5.0), velocity=(0.0, 0.0), embedding=look(0))
    weak = walker(2, "cam_1", 11.2, 20.0, (14.1, 5.0), velocity=(0.0, 0.0), embedding=look(80.0))
    other = walker(2, "cam_1", 11.2, 20.0, (14.1, 5.0), velocity=(0.0, 0.0), embedding=CLEARLY_OTHER)
    farther = walker(2, "cam_1", 11.2, 20.0, (15.2, 5.0), velocity=(0.0, 0.0), embedding=look(80.0))
    zones = ZoneMap([Zone("DOOR", 14.0, 5.0, 1.5)])
    assert groups(run([a, weak], zones)) == {frozenset({1, 2})}
    assert groups(run([a, other], zones)) == {frozenset({1}), frozenset({2})}
    assert groups(run([a, farther], zones)) == {frozenset({1}), frozenset({2})}


def test_standing_person_may_return_after_20s_in_place() -> None:
    a = walker(1, "cam_1", 0.0, 10.0, (5.0, 5.0), velocity=(0.0, 0.0), embedding=look(0))
    b = walker(2, "cam_1", 30.0, 40.0, (5.1, 5.0), velocity=(0.0, 0.0), embedding=SAME)
    far = walker(2, "cam_1", 30.0, 40.0, (9.0, 5.0), velocity=(0.0, 0.0), embedding=SAME)
    assert groups(run([a, b])) == {frozenset({1, 2})}
    assert groups(run([a, far])) == {frozenset({1}), frozenset({2})}


def test_impossible_speed_is_never_linked() -> None:
    a = walker(1, "cam_1", 0.0, 10.0, (0.0, 5.0))
    b = walker(2, "cam_2", 11.0, 20.0, (40.0, 5.0))
    assert groups(run([a, b])) == {frozenset({1}), frozenset({2})}


def test_blind_passage_depends_on_strict_setting() -> None:
    a = walker(1, "cam_1", 0.0, 10.0, (0.0, 5.0), embedding=look(0), ends_at_edge=True)
    b = walker(2, "cam_2", 13.0, 20.0, (18.0, 5.0), embedding=SAME, starts_at_edge=True)
    link = Zone("EDGE_LINK", 14.0, 5.0, 1.0, to_x=18.0, to_y=5.0, to_r=1.5)
    result = run([a, b], ZoneMap([link]))
    assert groups(result) == {frozenset({1, 2})}
    assert [s.kind for s in result.tracks[0].segments] == ["observed", "blind_inferred", "observed"]
    assert groups(run([a, b], ZoneMap([link], strict_blind=True))) == {frozenset({1}), frozenset({2})}


def test_edge_exit_needs_matching_appearance() -> None:
    a = walker(1, "cam_1", 0.0, 10.0, (0.0, 5.0), embedding=look(0), ends_at_edge=True)
    same = walker(2, "cam_1", 11.0, 20.0, (14.5, 5.0), embedding=SAME)
    other = walker(2, "cam_1", 11.0, 20.0, (14.5, 5.0), embedding=CLEARLY_OTHER)
    assert groups(run([a, same])) == {frozenset({1, 2})}
    assert groups(run([a, other])) == {frozenset({1}), frozenset({2})}


# Устойчивое перекрытие и слияние


def test_sustained_overlap_links_without_appearance() -> None:
    a = walker(1, "cam_1", 0.0, 10.0, (0.0, 5.0))
    b = walker(2, "cam_2", 8.0, 20.0, (11.2, 5.0))
    result = run([a, b])
    assert groups(result) == {frozenset({1, 2})}
    link = result.tracks[0].links[0]
    assert link.tier == "sustained"
    assert link.confidence == "green"
    assert link.contact_s == pytest.approx(2.0)


def test_track_stays_open_while_visible_on_another_camera() -> None:
    a = walker(1, "cam_1", 0.0, 10.0, (0.0, 5.0))
    b = walker(2, "cam_2", 5.0, 30.0, (7.0, 5.0))
    result = run([a, b])
    assert len(result.tracks) == 1
    assert [s.kind for s in result.tracks[0].segments] == ["observed", "observed"]


def test_competing_candidate_with_other_appearance_loses() -> None:
    # Двое рядом на второй камере, но первый из них выглядит иначе
    a = walker(1, "cam_1", 0.0, 10.0, (0.0, 5.0), embedding=look(0))
    wrong = walker(2, "cam_2", 8.0, 20.0, (11.2, 5.1), embedding=CLEARLY_OTHER)
    right = walker(3, "cam_2", 8.0, 20.0, (11.2, 5.4), embedding=SAME)
    assert groups(run([a, wrong, right])) == {frozenset({1, 3}), frozenset({2})}


def test_three_cameras_chain_into_one_track() -> None:
    a = walker(1, "cam_1", 0.0, 10.0, (0.0, 5.0))
    b = walker(2, "cam_2", 8.0, 20.0, (11.2, 5.0), ends_at_edge=True)
    c = walker(3, "cam_3", 19.9, 30.0, (27.86, 5.0), starts_at_edge=True)
    result = run([a, b, c])
    assert groups(result) == {frozenset({1, 2, 3})}
    assert [link.tier for link in result.tracks[0].links] == ["sustained", "brief"]


def test_short_tracklet_bridges_cameras_and_is_not_dropped_early() -> None:
    a = walker(1, "cam_1", 0.0, 10.0, (0.0, 5.0), ends_at_edge=True)
    bridge = walker(2, "cam_2", 10.1, 10.5, (14.14, 5.0), starts_at_edge=True, ends_at_edge=True)
    c = walker(3, "cam_3", 10.6, 20.0, (14.84, 5.0), starts_at_edge=True)
    assert groups(run([a, bridge, c])) == {frozenset({1, 2, 3})}


def test_brief_contact_cannot_take_person_from_sustained_handoff() -> None:
    a = walker(1, "cam_1", 0.0, 10.0, (0.0, 5.0), ends_at_edge=True)
    b = walker(2, "cam_2", 8.0, 20.0, (11.2, 5.0))
    c = walker(3, "cam_2", 9.9, 20.0, (13.86, 5.2), starts_at_edge=True)
    assert groups(run([a, b, c])) == {frozenset({1, 2}), frozenset({3})}


def test_group_with_contradiction_is_not_merged() -> None:
    # Камера со слабой точностью видит одного человека рядом с обоими, но эти двое в 3 м друг от друга
    a = walker(1, "cam_1", 0.0, 20.0, (0.0, 5.0))
    blurred = walker(2, "cam_2", 5.0, 15.0, (7.0, 6.5), sigma=1.5)
    c = walker(3, "cam_3", 0.0, 20.0, (0.0, 8.0))
    result = run([a, blurred, c])
    assert len(result.tracks) == 2
    assert frozenset({1, 3}) not in groups(result)


def test_ambiguous_brief_contact_is_not_merged() -> None:
    a = walker(1, "cam_1", 0.0, 10.0, (0.0, 5.0), embedding=look(0), ends_at_edge=True)
    b1 = walker(2, "cam_2", 10.0, 20.0, (14.0, 5.0), embedding=look(0), starts_at_edge=True)
    b2 = walker(3, "cam_2", 10.0, 20.0, (14.0, 5.1), embedding=look(0), starts_at_edge=True)
    assert groups(run([a, b1, b2])) == {frozenset({1}), frozenset({2}), frozenset({3})}


# Результат


def test_result_does_not_depend_on_input_order() -> None:
    tracklets = [
        walker(1, "cam_1", 0.0, 10.0, (0.0, 5.0)),
        walker(2, "cam_2", 8.0, 20.0, (11.2, 5.0)),
        walker(3, "cam_1", 50.0, 60.0, (0.0, 9.0)),
        walker(4, "cam_3", 55.0, 70.0, (7.0, 9.0)),
    ]
    forward = run(tracklets)
    backward = run(tracklets[::-1])
    assert [(t.track_id, t.tracklets) for t in forward.tracks] == [(t.track_id, t.tracklets) for t in backward.tracks]
    assert [t.track_id for t in forward.tracks] == [1, 2]
    assert forward.tracks[0].tracklets == [1, 2]


def test_short_final_track_is_dropped_from_count() -> None:
    a = walker(1, "cam_1", 0.0, 10.0, (0.0, 5.0))
    flash = walker(2, "cam_1", 30.0, 30.3, (20.0, 20.0))
    result = run([a, flash])
    assert groups(result) == {frozenset({1})}
    assert result.dropped == [2]


def test_hint_goes_to_one_new_track_and_never_changes_count() -> None:
    gone = walker(1, "cam_1", 0.0, 10.0, (0.0, 5.0), embedding=look(0))
    first = walker(2, "cam_1", 100.0, 110.0, (0.0, 5.0), embedding=SAME)
    second = walker(3, "cam_1", 200.0, 210.0, (0.0, 5.0), embedding=look(20.0))
    result = run([gone, first, second])
    assert len(result.tracks) == 3
    hinted = [t for t in result.tracks if t.hints and t.hints[0].track_id == result.tracks[0].track_id]
    assert [t.tracklets for t in hinted] == [[2]]


def test_hint_needs_margin_over_second_candidate() -> None:
    one = walker(1, "cam_1", 0.0, 10.0, (0.0, 5.0), embedding=look(0))
    twin = walker(2, "cam_1", 0.0, 10.0, (0.0, 9.0), embedding=look(1.0))
    back = walker(3, "cam_1", 100.0, 110.0, (0.0, 5.0), embedding=look(0))
    result = run([one, twin, back])
    assert all(not t.hints for t in result.tracks)


def test_hint_needs_strong_similarity() -> None:
    gone = walker(1, "cam_1", 0.0, 10.0, (0.0, 5.0), embedding=look(0))
    other = walker(2, "cam_1", 100.0, 110.0, (0.0, 5.0), embedding=look(60.0))
    assert all(not t.hints for t in run([gone, other]).tracks)


def test_invariants_catch_merged_overlap_on_one_camera() -> None:
    a = walker(1, "cam_1", 0.0, 10.0, (0.0, 5.0))
    b = walker(2, "cam_1", 5.0, 15.0, (7.0, 5.0))
    result = run([a, b])
    result.tracks[0].tracklets = [1, 2]
    assert any("одной камеры" in problem for problem in check_invariants(result, [a, b], CONFIG))


def test_tracker_handover_with_short_overlap_on_one_camera_is_one_person() -> None:
    a = walker(1, "cam_1", 0.0, 10.0, (0.0, 5.0))
    b = walker(2, "cam_1", 9.8, 20.0, (13.72, 5.0))
    assert groups(run([a, b])) == {frozenset({1, 2})}


def test_short_overlap_of_two_distant_people_on_one_camera_stays_apart() -> None:
    a = walker(1, "cam_1", 0.0, 10.0, (0.0, 5.0))
    b = walker(2, "cam_1", 9.8, 20.0, (13.72, 9.0))
    assert groups(run([a, b])) == {frozenset({1}), frozenset({2})}


def test_handover_on_one_camera_does_not_block_link_with_other_camera() -> None:
    # Один человек на второй камере виден всё время, а первая камера сменила номер с коротким пересечением
    a = walker(1, "cam_1", 0.0, 10.0, (0.0, 5.0))
    b = walker(2, "cam_1", 9.8, 20.0, (13.72, 5.0))
    other = walker(3, "cam_2", 2.0, 18.0, (2.8, 5.0))
    assert groups(run([a, b, other])) == {frozenset({1, 2, 3})}


# Обломки: короткие детекции человека, которого уже уверенно видит другая камера


def test_short_fragment_inside_long_track_is_absorbed() -> None:
    a = walker(1, "cam_1", 0.0, 20.0, (0.0, 5.0))
    fragment = walker(2, "cam_2", 10.0, 10.4, (14.0, 5.0))
    result = run([a, fragment])
    assert groups(result) == {frozenset({1, 2})}
    assert result.tracks[0].links[0].tier == "brief"


def test_short_fragment_far_from_track_is_not_absorbed() -> None:
    a = walker(1, "cam_1", 0.0, 20.0, (0.0, 5.0))
    fragment = walker(2, "cam_2", 10.0, 10.4, (14.0, 12.0))
    result = run([a, fragment])
    assert groups(result) == {frozenset({1})}
    assert result.dropped == [2]


def test_short_fragment_with_huge_error_needs_small_distance() -> None:
    a = walker(1, "cam_1", 0.0, 20.0, (0.0, 5.0))
    fragment = walker(2, "cam_2", 10.0, 10.4, (14.0, 9.5), sigma=4.0)
    result = run([a, fragment])
    assert groups(result) == {frozenset({1})}


def test_short_fragment_between_two_people_stays_unlinked() -> None:
    a1 = walker(1, "cam_1", 0.0, 20.0, (0.0, 5.0))
    a2 = walker(2, "cam_1", 0.0, 20.0, (0.0, 5.5))
    fragment = walker(3, "cam_2", 10.0, 10.4, (14.0, 5.25))
    result = run([a1, a2, fragment])
    assert groups(result) == {frozenset({1}), frozenset({2})}
    assert result.dropped == [3]


def test_track_from_fragments_is_dropped_by_observed_time_not_span() -> None:
    first = walker(1, "cam_1", 0.0, 0.3, (0.0, 5.0))
    second = walker(2, "cam_1", 1.9, 2.2, (2.7, 5.0))
    result = run([first, second])
    assert result.tracks == []
    assert result.dropped == [1, 2]


def test_hint_needs_physically_reachable_return() -> None:
    gone = walker(1, "cam_1", 0.0, 10.0, (0.0, 5.0), embedding=look(0))
    too_far = walker(2, "cam_2", 11.5, 21.5, (50.0, 5.0), embedding=SAME)
    assert all(not t.hints for t in run([gone, too_far]).tracks)
