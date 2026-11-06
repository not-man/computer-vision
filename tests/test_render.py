import json
import subprocess
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

from mct.geometry import IMAGE_SIZE
from mct.render import (
    CELL,
    CameraStream,
    Silhouette,
    Stamps,
    TrackView,
    build_events,
    assign_colors,
    chunk_index,
    draw_tag,
    paint_silhouette,
    ticks,
)
from mct.video import FFMPEG, H264Writer, clock_text, read_frames


def fake_frames(numbers: list[int]):
    for number in numbers:
        yield number, np.full((4, 4, 3), number % 256, np.uint8)


def track_json(track_id: int = 7, **overrides) -> dict:
    track = {
        "track_id": track_id, "t_start_ms": 10_000, "t_end_ms": 40_000, "close_zone": "DOOR",
        "segments": [
            {"kind": "observed", "camera_id": "camera_new_1", "t_start_ms": 10_000, "t_end_ms": 25_000},
            {"kind": "observed", "camera_id": "camera_new_2", "t_start_ms": 20_000, "t_end_ms": 40_000},
        ],
        "links": [{"a": 1, "b": 2, "tier": "sustained", "t_ms": 20_000, "g_m": 0.4, "contact_s": 5.0, "r": 0.71,
                   "score": 0.8, "confidence": "green", "camera_a": "camera_new_1", "camera_b": "camera_new_2"}],
        "plan_path": [[1.0, 1.0, 10_000], [2.0, 1.0, 10_500], [3.0, 1.0, 11_000], [9.0, 1.0, 15_000], [10.0, 1.0, 15_500]],
        "possibly_same_person_as": [],
    }
    track.update(overrides)
    return track


# Синхронность


def test_stream_returns_nearest_frame_by_common_clock() -> None:
    # 25 кадр/с, выборка 10 кадр/с: номера 0, 3, 5, 8, 10; часы начинаются с 100 с
    stream = CameraStream(fake_frames([0, 3, 5, 8, 10]), 100.0, 25.0)
    assert stream.at(100.0)[0] == 0
    assert stream.at(100.1)[0] == 3
    assert stream.at(100.2)[0] == 5
    assert stream.at(100.3)[0] == 8
    assert stream.at(100.4)[0] == 10


def test_stream_shows_nothing_before_and_after_recording() -> None:
    stream = CameraStream(fake_frames([0, 5]), 100.0, 25.0)
    assert stream.at(90.0) is None
    assert stream.at(100.0)[0] == 0
    assert stream.at(100.2)[0] == 5
    assert stream.at(101.0) is None


def test_cameras_with_different_start_stay_within_half_a_step() -> None:
    late = CameraStream(fake_frames(list(range(0, 500, 2))), 104.5, 20.0)
    early = CameraStream(fake_frames([int(np.ceil(k * 2.5 - 1e-9)) for k in range(400)]), 100.3, 25.0)
    for t in ticks(105.0, 115.0):
        a, b = late.at(float(t)), early.at(float(t))
        assert abs(104.5 + a[0] / 20.0 - t) <= 0.051
        assert abs(100.3 + b[0] / 25.0 - t) <= 0.051


def test_ticks_and_chunks_are_exact() -> None:
    times = ticks(100.0, 220.0)
    assert len(times) == 1200
    assert chunk_index(times[0], 100.0, 60) == 0
    assert chunk_index(times[599], 100.0, 60) == 0
    assert chunk_index(times[600], 100.0, 60) == 1
    assert chunk_index(times[-1], 100.0, 60) == 1


# Силуэт


def silhouette(track: int | None = 1) -> Silhouette:
    width, height = IMAGE_SIZE
    mask = np.zeros((height, width), np.uint8)
    mask[300:600, 400:500] = 1
    rle = mask_utils.encode(np.asfortranarray(mask))["counts"].decode("ascii")
    return Silhouette(track, rle, (400.0, 300.0, 500.0, 600.0))


def test_mask_lands_at_scaled_position_and_leaves_the_rest_untouched() -> None:
    cell = np.full((CELL[1], CELL[0], 3), 100, np.uint8)
    rectangle = paint_silhouette(cell, silhouette(), (0, 0, 255))
    left, top, width, height = rectangle
    assert abs(left - 400 * CELL[0] / IMAGE_SIZE[0]) <= 1
    assert abs(top - 300 * CELL[1] / IMAGE_SIZE[1]) <= 1
    assert abs(width - 100 * CELL[0] / IMAGE_SIZE[0]) <= 1
    assert abs(height - 300 * CELL[1] / IMAGE_SIZE[1]) <= 1
    center = cell[top + height // 2, left + width // 2]
    assert center[2] > center[0]
    assert (cell[0, 0] == 100).all()
    outside = cell.copy()
    outside[top:top + height, left:left + width] = 100
    assert (outside == 100).all()


def test_fragment_gets_contour_without_fill() -> None:
    cell = np.full((CELL[1], CELL[0], 3), 100, np.uint8)
    left, top, width, height = paint_silhouette(cell, silhouette(None), None)
    assert (cell[top + height // 2, left + width // 2] == 100).all()
    assert (cell[top:top + height, left:left + width] != 100).any()


def test_mask_at_frame_corner_does_not_fail() -> None:
    width, height = IMAGE_SIZE
    mask = np.zeros((height, width), np.uint8)
    mask[height - 50:, width - 50:] = 1
    rle = mask_utils.encode(np.asfortranarray(mask))["counts"].decode("ascii")
    cell = np.zeros((CELL[1], CELL[0], 3), np.uint8)
    assert paint_silhouette(cell, Silhouette(1, rle, (width - 50.0, height - 50.0, width + 5.0, height + 5.0)), (0, 255, 0)) is not None


def span(track_id: int, start_s: float, end_s: float) -> dict:
    return {"track_id": track_id, "t_start_ms": int(start_s * 1000), "t_end_ms": int(end_s * 1000)}


def test_simultaneous_tracks_never_share_a_color() -> None:
    tracks = [span(i, 0.0, 100.0) for i in range(1, 11)] + [span(11, 50.0, 60.0)]
    colors = assign_colors(tracks)
    assert len(set(colors.values())) == len(tracks)
    assert colors == assign_colors(tracks[::-1])


def test_color_returns_only_after_the_trail_is_gone() -> None:
    tracks = [span(i, 0.0, 10.0) for i in range(1, 13)] + [span(13, 11.0, 30.0), span(14, 100.0, 110.0)]
    colors = assign_colors(tracks)
    # Вся палитра занята, поэтому номеру 13 достаётся цвет, но палитра не заканчивается ошибкой
    assert colors[13] in colors.values()
    assert colors[14] != colors[13]
    assert len(set(colors[i] for i in range(1, 13))) == 12


# Живой план


def test_position_interpolates_and_stays_at_last_point_during_pause() -> None:
    view = TrackView.from_json(track_json())
    assert view.position(9.0) is None
    position, hidden = view.position(10.25)
    assert not hidden and np.allclose(position, (1.5, 1.0))
    position, hidden = view.position(13.0)
    assert hidden and np.allclose(position, (3.0, 1.0))
    assert view.position(16.0) is None


def test_trail_is_dashed_only_across_the_pause() -> None:
    view = TrackView.from_json(track_json())
    pieces = view.trail(15.5, length_s=10.0)
    assert [pause for _, pause in pieces] == [None, "occluded", None]
    assert np.allclose(pieces[1][0], [[3.0, 1.0], [9.0, 1.0]])


def test_dash_kind_comes_from_the_track_segment() -> None:
    segments = track_json()["segments"] + [{"kind": "blind_inferred", "camera_id": None, "t_start_ms": 11_000, "t_end_ms": 15_000}]
    view = TrackView.from_json(track_json(segments=segments))
    assert [pause for _, pause in view.trail(15.5, length_s=10.0)] == [None, "blind_inferred", None]


def test_cameras_now_during_pause_is_empty() -> None:
    segments = [{"kind": "observed", "camera_id": "camera_new_1", "t_start_ms": 10_000, "t_end_ms": 11_000},
                {"kind": "occluded", "camera_id": None, "t_start_ms": 11_000, "t_end_ms": 15_000},
                {"kind": "observed", "camera_id": "camera_new_1", "t_start_ms": 15_000, "t_end_ms": 15_500}]
    view = TrackView.from_json(track_json(segments=segments))
    assert view.cameras_now(13.0) == []
    assert view.cameras_now(15.2) == ["1"]


def test_cameras_now_follow_segments() -> None:
    view = TrackView.from_json(track_json())
    assert view.cameras_now(12.0) == ["1"]
    assert view.cameras_now(22.0) == ["1", "2"]
    assert view.cameras_now(30.0) == ["2"]


# Плашки


def test_events_describe_handoff_and_closure() -> None:
    events = build_events({"tracks": [track_json()]})
    assert [e.kind for e in events] == ["handoff", "close"]
    assert "cam_1 → cam_2" in events[0].text and "G 0.4" in events[0].text and "R 0.71" in events[0].text
    assert "закрыт" in events[1].text and "дверь" in events[1].text
    assert events[1].t == 40.0


def test_yellow_link_is_marked_and_tracks_open_at_recording_end_are_not_closed() -> None:
    link = dict(track_json()["links"][0], confidence="yellow")
    events = build_events({"tracks": [track_json(links=[link])]}, recording_end_s=40.5)
    assert [e.kind for e in events] == ["handoff"]
    assert "спорная" in events[0].text


def test_ghost_plate_keeps_the_wording_from_the_design() -> None:
    hint = [{"track_id": 7, "r": 0.81, "absent_s": 134.0}]
    events = build_events({"tracks": [track_json(track_id=12, links=[], possibly_same_person_as=hint)]})
    ghost = next(e for e in events if e.kind == "ghost")
    assert ghost.text == "НОВЫЙ ТРЕК #12 · Вероятно, вернулся #7 · отсутствовал 2:14 · сходство 0.81 · не склеено по правилу ТЗ"
    assert ghost.old_track_id == 7


def test_fast_and_gap_links_have_their_own_plates() -> None:
    fast = dict(track_json()["links"][0], tier="brief", contact_s=-0.3)
    gap = dict(track_json()["links"][0], tier="gap", contact_s=-3.0, camera_b="camera_new_1")
    events = build_events({"tracks": [track_json(links=[fast, gap])]})
    assert any(e.kind == "fast" and "разрыв 0.3" in e.text for e in events)
    assert any(e.kind == "resume" and "скрыт 3.0" in e.text and "на cam_1" in e.text for e in events)


def test_adjacent_gap_is_not_described_as_hidden() -> None:
    adjacent = dict(track_json()["links"][0], tier="gap", contact_s=-0.0)
    text = build_events({"tracks": [track_json(links=[adjacent])]})[0].text
    assert "встык" in text and "скрыт" not in text


def test_cyrillic_stamp_is_cached_and_wraps() -> None:
    stamps = Stamps()
    first = stamps.text("Треков по ТЗ", 14)
    assert first is stamps.text("Треков по ТЗ", 14)
    assert (first != first[0, 0]).any()
    wrapped = stamps.text("очень длинная надпись на плашке события которая не влезает", 14, max_width=120)
    assert wrapped.shape[0] > first.shape[0]


def test_clock_text() -> None:
    assert clock_text(2502.4) == "17:41:42.4"
    # Секунды не округляются вверх до 60
    assert clock_text(2519.96) == "17:41:59.9"
    assert clock_text(2520.0) == "17:42:00.0"


def test_tags_of_people_side_by_side_do_not_cover_each_other() -> None:
    image = np.zeros((200, 200, 3), np.uint8)
    taken: list[tuple[int, int, int, int]] = []
    draw_tag(image, "#24", 50, 100, (0, 255, 0), taken)
    draw_tag(image, "#25", 55, 100, (0, 0, 255), taken)
    (x1, y1, w1, h1), (x2, y2, w2, h2) = taken
    assert x1 + w1 <= x2 or x2 + w2 <= x1


def test_tag_stays_inside_the_image() -> None:
    image = np.zeros((100, 100, 3), np.uint8)
    taken: list[tuple[int, int, int, int]] = []
    draw_tag(image, "#101", 95, 3, (0, 255, 0), taken)
    x, y, w, h = taken[0]
    assert 0 <= x and x + w <= 100 and 0 <= y and y + h <= 100


# Файл, который играет браузер


def test_chunk_is_h264_with_faststart_and_reads_back(tmp_path: Path) -> None:
    path = tmp_path / "chunk.mp4"
    with H264Writer(path, (320, 240), 10.0) as writer:
        for value in range(30):
            frame = np.full((240, 320, 3), value * 8, np.uint8)
            writer.write(frame)
    header = subprocess.run([FFMPEG, "-hide_banner", "-i", str(path)], capture_output=True, text=True).stderr
    assert "h264" in header and "yuv420p" in header
    assert b"moov" in path.read_bytes()[:4096]
    frames = list(read_frames(path, (320, 240), 10.0, 10.0))
    assert len(frames) == 30
    assert frames[0][1].mean() < frames[-1][1].mean()


def test_manifest_is_json_friendly() -> None:
    events = build_events({"tracks": [track_json()]})
    json.dumps([{"t": e.t, "text": e.text} for e in events], ensure_ascii=False)
