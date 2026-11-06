import numpy as np

from pathlib import Path

from mct.video import H264Writer, read_frames, read_selected_frames, sampled_frame_index


def selected_by_filter(step: float, frames: int) -> list[int]:
    # То же условие, что в фильтре select из read_frames
    return [n for n in range(frames) if np.mod(n, step) < 1]


def test_sampled_index_matches_select_filter() -> None:
    for step in (2.5, 2.0):
        kept = selected_by_filter(step, 1000)
        assert [sampled_frame_index(k, step) for k in range(len(kept))] == kept


def test_sampling_rate_is_ten_per_clock_second() -> None:
    assert len(selected_by_filter(2.5, 250)) == 100
    assert len(selected_by_filter(2.0, 200)) == 100


def test_start_offset_returns_the_same_frames_and_numbers(tmp_path: Path) -> None:
    path = tmp_path / "numbers.mp4"
    with H264Writer(path, (64, 48), 25.0) as writer:
        for value in range(100):
            writer.write(np.full((48, 64, 3), value * 2, np.uint8))
    full = list(read_frames(path, (64, 48), 25.0, 10.0))
    part = list(read_frames(path, (64, 48), 25.0, 10.0, start_s=2.0, duration_s=1.0))
    assert [n for n, _ in full][:5] == [0, 3, 5, 8, 10]
    assert len(part) == 10
    assert [n for n, _ in part] == [n for n, _ in full][20:30]
    assert all(np.array_equal(a, b) for (_, a), (_, b) in zip(part, full[20:30]))


def test_selected_frames_have_the_same_numbers_as_sequential_reading(tmp_path: Path) -> None:
    path = tmp_path / "numbers.mp4"
    with H264Writer(path, (64, 48), 25.0) as writer:
        for value in range(100):
            writer.write(np.full((48, 64, 3), value * 2, np.uint8))
    full = dict(read_frames(path, (64, 48), 25.0, 25.0))
    picked = list(read_selected_frames(path, [80, 3, 40, 3], (64, 48)))
    assert [number for number, _ in picked] == [3, 40, 80]
    assert all(abs(float(frame.mean()) - float(full[number].mean())) < 2.0 for number, frame in picked)


def test_selected_frames_stop_at_the_end_of_the_file(tmp_path: Path) -> None:
    path = tmp_path / "short.mp4"
    with H264Writer(path, (64, 48), 25.0) as writer:
        for value in range(10):
            writer.write(np.full((48, 64, 3), value, np.uint8))
    assert [number for number, _ in read_selected_frames(path, [5, 500], (64, 48))] == [5]
