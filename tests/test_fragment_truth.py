import numpy as np
import pandas as pd

from mct.embedder import without_background
from mct.fragment_truth import DIFFERENT_PEOPLE, LABELS, SAME_PERSON, check_pairs, label_rows, score


def rows_for_labels() -> pd.DataFrame:
    return pd.DataFrame([
        {"camera": camera, "raw_track": raw_track, "t_s": t}
        for camera, raw_track, start, end, _ in LABELS for t in (start, (start + end) / 2, end)
    ])


def test_labels_cover_their_own_intervals_and_nothing_else() -> None:
    rows = rows_for_labels()
    assert label_rows(rows).notna().all()
    outside = rows.assign(t_s=rows.t_s + 1000.0)
    assert label_rows(outside).isna().all()


def test_perfect_tracks_score_one() -> None:
    rows = rows_for_labels()
    people = label_rows(rows)
    numbers = score(rows, people.map({person: number for number, person in enumerate(sorted(set(people)))}))
    assert numbers["соответствие"] == 1.0 and numbers["чистота"] == 1.0 and numbers["людей"] == numbers["треков"]


def test_one_track_for_everybody_is_not_pure() -> None:
    rows = rows_for_labels()
    assert score(rows, pd.Series(1, index=rows.index))["чистота"] < 0.5


def probe_rows() -> pd.DataFrame:
    points = {point for a, b, _ in SAME_PERSON + DIFFERENT_PEOPLE for point in (a, b)}
    return pd.DataFrame([{"camera": c, "raw_track": r, "t_s": t} for c, r, t in sorted(points)])


def test_pairs_report_what_is_wrong() -> None:
    rows = probe_rows()
    separately = pd.Series(np.arange(len(rows)), index=rows.index)
    errors = check_pairs(rows, separately)
    assert len(errors["должны быть вместе, но врозь"]) == len(SAME_PERSON) and errors["должны быть врозь, но вместе"] == []
    together = check_pairs(rows, pd.Series(1, index=rows.index))
    assert together["должны быть вместе, но врозь"] == [] and len(together["должны быть врозь, но вместе"]) == len(DIFFERENT_PEOPLE)


def test_background_is_painted_outside_the_mask() -> None:
    crop = np.full((20, 10, 3), 200, np.uint8)
    mask = np.zeros((20, 10), np.uint8)
    mask[8:12, 4:6] = 1
    painted = without_background(crop, mask)
    assert (painted[10, 5] == 200).all()
    assert (painted[0, 0] == (104, 116, 124)).all()
