import numpy as np
import pytest

from mct.sync import PairLag, camera_offsets, pair_lag, summarize_pair

T = np.arange(0.0, 8.0, 0.1)


def walk(t: np.ndarray, speed: float = 1.2, bias: float = 0.0) -> np.ndarray:
    return np.column_stack([np.full(len(t), 5.0), 2.0 + speed * t + bias])


def test_known_lag_is_recovered() -> None:
    # Часы a опережают часы b на 0.4 с: та же точка на a записана с временем на 0.4 с больше
    found = pair_lag("a", T + 0.4, walk(T), "b", T, walk(T))
    assert found.lag == pytest.approx(0.4, abs=0.051)
    assert found.direction == "y+"
    assert found.distance < 0.05 < found.distance_at_zero


def test_standing_person_says_nothing_about_time() -> None:
    assert pair_lag("a", T, walk(T, speed=0.0), "b", T, walk(T, speed=0.0)) is None


def test_short_overlap_says_nothing() -> None:
    assert pair_lag("a", T, walk(T), "b", T + 7.5, walk(T)) is None


def test_position_error_cancels_between_opposite_directions() -> None:
    # Настоящий сдвиг 0.3 с и ошибка положения 0.36 м: для одного потока они гасят друг друга, для встречного складываются
    forward = pair_lag("a", T + 0.3, walk(T), "b", T, walk(T, bias=-0.36))
    backward = pair_lag("a", T + 0.3, walk(T, speed=-1.2), "b", T, walk(T, speed=-1.2, bias=-0.36))
    assert forward.lag == pytest.approx(0.0, abs=0.06)
    assert backward.lag == pytest.approx(0.6, abs=0.06)
    lag, _ = summarize_pair([forward, forward, backward, backward])
    assert lag == pytest.approx(0.3, abs=0.06)


def lag_item(a: str, b: str, lag: float, direction: str = "y+") -> PairLag:
    return PairLag(a, b, direction, lag, 0.1, 0.5, 1.2)


def test_offsets_are_solved_over_a_chain_of_cameras() -> None:
    truth = {"1": 0.4, "2": -0.1, "3": -0.3}
    items = []
    for a, b in (("1", "2"), ("2", "3")):
        for direction in ("y+", "y+", "y-", "y-"):
            items.append(lag_item(a, b, truth[a] - truth[b], direction))
    offsets, residuals = camera_offsets(items, ["1", "2", "3"])
    assert offsets == pytest.approx(truth, abs=1e-6)
    assert all(abs(value) < 1e-6 for value in residuals.values())


def test_one_direction_counts_less_than_both() -> None:
    both = [lag_item("1", "2", 0.2, "y+"), lag_item("1", "2", 0.2, "y+"), lag_item("1", "2", 0.2, "y-"), lag_item("1", "2", 0.2, "y-")]
    one = [lag_item("1", "2", 0.2, "y+")] * 4
    assert summarize_pair(both)[1] > summarize_pair(one)[1]
