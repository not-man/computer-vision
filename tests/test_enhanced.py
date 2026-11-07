from types import SimpleNamespace
from pathlib import Path

import numpy as np
import yaml

from mct.enhanced import Detection, active_regions, deduplicate, project_detection


def detection(box, origin, score=.8, clipped=False):
    mask = np.zeros((100, 100), bool)
    x1, y1, x2, y2 = box
    mask[y1:y2, x1:x2] = True
    return Detection(np.array(box), score, mask, origin, clipped)


def test_roi_coordinates_and_mask_match_existing_geometry():
    item = project_detection([10, 20, 30, 60], .8, np.ones((100, 100)), (200, 100, 100, 100), (1000, 800), (500, 400), "roi")
    np.testing.assert_allclose(item.box, [105, 60, 115, 80])
    assert item.mask.shape == (400, 500)
    assert item.mask[50:100, 100:150].all()
    assert item.mask.sum() == 2500


def test_duplicate_from_full_frame_and_roi_is_removed():
    assert len(deduplicate([detection([10, 10, 40, 90], "full"), detection([11, 10, 41, 90], "roi")])) == 1


def test_close_people_in_same_pass_are_preserved():
    assert len(deduplicate([detection([10, 10, 40, 90], "full"), detection([30, 10, 60, 90], "full")])) == 2


def test_two_separate_people_replace_merged_mask():
    assert len(deduplicate([detection([10, 10, 90, 90], "full"), detection([10, 10, 45, 90], "roi"),
                            detection([55, 10, 90, 90], "roi")])) == 2


def test_head_or_torso_fragment_does_not_create_second_id():
    for origin in ("roi", "full"):
        assert len(deduplicate([detection([10, 10, 40, 90], "full"), detection([15, 10, 35, 30], origin)])) == 1


def test_crop_edge_fragment_is_not_sent_to_tracker():
    assert deduplicate([detection([10, 10, 40, 90], "roi", clipped=True)]) == []


def test_head_and_body_parts_do_not_masquerade_as_two_people():
    assert len(deduplicate([detection([10, 10, 40, 90], "full"), detection([10, 10, 40, 30], "roi"),
                            detection([10, 30, 40, 90], "roi")])) == 1


def test_complete_mask_preferred_over_clipped_high_score():
    full = detection([10, 10, 40, 90], "full", .7)
    clipped = detection([10, 10, 40, 90], "roi", .95, True)
    assert deduplicate([clipped, full])[0] is full


def test_roi_windows_are_half_open_and_disabled_by_default():
    cfg = {"regions": {"cam": [{"enabled": True, "windows": [[10, 20]]}, {}]}}
    assert len(active_regions(cfg, "cam", 10)) == 1
    assert not active_regions(cfg, "cam", 20)
    assert not active_regions(cfg, "other", 15)


def test_tracker_indices_still_address_input_masks_with_empty_frames():
    from ultralytics.engine.results import Boxes
    from ultralytics.trackers.bot_sort import BOTSORT
    tracker = BOTSORT(SimpleNamespace(**yaml.safe_load(Path("configs/botsort.yaml").read_text())))
    frame = np.zeros((100, 100, 3), np.uint8)
    data = np.array([[10, 10, 30, 70, .9, 0], [60, 10, 80, 70, .85, 0]], np.float32)
    tracks = tracker.update(Boxes(data, (100, 100)).numpy(), frame)
    assert sorted(tracks[:, -1].astype(int)) == [0, 1]
    tracker.update(Boxes(np.empty((0, 6), np.float32), (100, 100)).numpy(), frame)
    tracks = tracker.update(Boxes(data, (100, 100)).numpy(), frame)
    assert sorted(tracks[:, -1].astype(int)) == [0, 1]


def test_pilot_cannot_be_linked_as_full_recording(tmp_path):
    from mct.tracklets import load_perception
    (tmp_path / "PILOT_ONLY").touch()
    import pytest
    with pytest.raises(ValueError, match="Pilot"):
        load_perception(tmp_path, {}, None)
