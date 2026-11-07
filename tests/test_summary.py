import json
import re
from pathlib import Path

import pandas as pd
from PIL import Image

from mct.report import build_report
from mct.summary import KNOWN_ERRORS, check_known_errors, hint_texts, link_rows, quality_tiles, rule_rows, segment_rows, track_rows, track_video_chunks


def track(track_id: int = 1, start_s: float = 0.0, end_s: float = 10.0) -> dict:
    return {
        "track_id": track_id, "t_start_ms": int(start_s * 1000), "t_end_ms": int(end_s * 1000), "start": "17:00:00.0", "end": "17:00:10.0",
        "cameras": ["camera_new_1", "camera_new_3"], "close_zone": "DOOR", "tracklets": [7],
        "segments": [{"kind": "observed", "camera_id": "camera_new_1", "t_start_ms": 0, "t_end_ms": 3000},
                     {"kind": "observed", "camera_id": "camera_new_1", "t_start_ms": 3500, "t_end_ms": 5000},
                     {"kind": "observed", "camera_id": "camera_new_3", "t_start_ms": 4000, "t_end_ms": 10000},
                     {"kind": "observed", "camera_id": "camera_new_1", "t_start_ms": 9000, "t_end_ms": 10000},
                     {"kind": "gap", "camera_id": None, "t_start_ms": 5000, "t_end_ms": 5500}],
        "links": [{"t_ms": 4000, "camera_a": "camera_new_1", "camera_b": "camera_new_3", "tier": "sustained", "g_m": 0.3, "contact_s": 1.0, "r": 0.5, "confidence": "yellow"}],
        "plan_path": [[10.0, 10.0, 0], [11.0, 10.0, 500]],
        "possibly_same_person_as": [{"track_id": 2, "r": 0.8, "absent_s": 317.0}],
    }


def test_segments_join_short_breaks_within_a_camera_and_keep_long_ones() -> None:
    rows = segment_rows(track())
    assert [(row["Камера"], row["С"], row["По"]) for row in rows] == [
        ("cam_1", "17:00:00", "17:00:05"), ("cam_3", "17:00:04", "17:00:10"), ("cam_1", "17:00:09", "17:00:10"),
    ]


def test_dossier_video_seeks_to_observation_in_each_minute() -> None:
    item = track(end_s=160.0)
    item["segments"] = [
        {"kind": "observed", "camera_id": "camera_new_1", "t_start_ms": 2000, "t_end_ms": 10000},
        {"kind": "observed", "camera_id": "camera_new_3", "t_start_ms": 145000, "t_end_ms": 155000},
    ]
    chunks = [{"file": f"chunks/{index:03}.mp4", "t_start_s": index * 60.0, "t_end_s": (index + 1) * 60.0,
               "start": f"17:0{index}:00.0"} for index in range(3)]
    chosen = track_video_chunks(item, chunks)
    assert [chunk["file"] for chunk in chosen] == ["chunks/000.mp4", "chunks/002.mp4"]
    assert chosen[1]["offset"] == 24.0
    assert chosen[1]["first_seen"] == "17:02:25"
    assert chosen[1]["cameras"] == "cam_3"


def test_track_table_has_what_the_brief_asks_for() -> None:
    row = track_rows([track()])[0]
    assert {"Трек", "Начало", "Конец", "Камеры"} <= set(row)
    assert row["Камеры"] == "1, 3"
    assert row["Похож на"] == "#2"


def test_doubtful_link_is_named_and_hint_shows_absence() -> None:
    assert link_rows(track())[0]["Оценка"] == "спорная"
    assert "5:17" in hint_texts(track())[0]


def test_measured_quality_is_shown_only_when_present() -> None:
    assert quality_tiles({}) == []
    tiles = quality_tiles({"quality": {"crowd": {"match": 0.926, "purity": 0.955, "people": 7, "tracks": 8}, "pairs": {"right": 25, "total": 28, "errors": {}}}})
    assert ("Пары, проверенные глазами: верно", "25 из 28") in tiles and tiles[0][1] == "0.93"
    assert len(tiles) == 3


def test_rules_are_sorted_from_the_most_harmful() -> None:
    robustness = pd.DataFrame({"name": ["мелочь", "важное"], "group": ["откл", "откл"], "tracks": [108, 213], "yellow": [37, 43], "agreement": [0.99, 0.86]})
    assert [row["Правило"] for row in rule_rows(robustness)] == ["важное", "мелочь"]


def test_known_errors_must_point_at_a_real_moment() -> None:
    error = KNOWN_ERRORS[0]
    good = {"track_id": error.track_id, "t_start_ms": int((error.t_s - 5) * 1000), "t_end_ms": int((error.t_s + 5) * 1000), "start": "", "end": ""}
    others = [{"track_id": other.track_id, "t_start_ms": int(other.t_s * 1000), "t_end_ms": int(other.t_s * 1000), "start": "", "end": ""} for other in KNOWN_ERRORS[1:]]
    assert check_known_errors([good, *others]) == []
    moved = {**good, "t_start_ms": 0, "t_end_ms": 1000}
    assert any(f"#{error.track_id}" in problem for problem in check_known_errors([moved, *others]))
    assert any(f"нет трека #{error.track_id}" in problem for problem in check_known_errors(others))


def test_report_needs_no_network_and_opens_videos_next_to_it(tmp_path: Path) -> None:
    tracks_dir = tmp_path / "tracks"
    (tracks_dir / "crops").mkdir(parents=True)
    Image.new("RGB", (40, 80), "white").save(tracks_dir / "crops" / "tracklet_7.jpg")
    (tracks_dir / "crops" / "index.json").write_text(json.dumps({"7": {"file": "tracklet_7.jpg", "camera": "camera_new_1", "t_s": 1.0}}), encoding="utf-8")
    plan = tmp_path / "plan.jpg"
    Image.new("RGB", (60, 60), "white").save(plan)
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "plan_features.yaml").write_text("plan_px_per_m: 29.0\nplan_image: {scale: 2.0, dx: 4.0, dy: 2.0}\n", encoding="utf-8")
    robustness = pd.DataFrame({
        "name": ["длительность 2.0", "r_veto 0.05", "правило"], "group": ["кривая длительности", "кривая r_veto", "откл"],
        "tracks": [108, 108, 120], "yellow": [37, 37, 40], "agreement": [1.0, 1.0, 0.9],
    })
    manifest = {"clock_start_s": 0.0, "clock_end_s": 60.0, "events": [{"t_s": 1.0, "text": "событие"}],
                "chunks": [{"file": "chunks/000.mp4", "t_start_s": 0.0, "t_end_s": 60.0, "start": "17:00:00.0"}]}
    out = tmp_path / "report.html"
    build_report({"total_tracks": 1, "tracks": [track()]}, manifest, robustness, tracks_dir, plan, configs, out)
    html = out.read_text(encoding="utf-8")
    data = json.loads(html[html.index("const DATA = ") + len("const DATA = "):html.index(";\nconst CONFIG")].replace("<\\/", "</"))
    assert data["tracks"]["1"]["chunks"][0] == {"label": "17:00:00", "file": "chunks/000.mp4", "offset": 0,
                                                 "first_seen": "17:00:00", "cameras": "cam_1, cam_3"}
    assert data["tracks"]["1"]["plan"]["layout"]["images"][0]["source"] == ""
    assert data["tracks"]["1"]["crops"][0]["uri"].startswith("data:image/jpeg;base64,")
    page_only = html[:html.index("<script>")]
    assert not re.search(r'(src|href)="https?:', page_only)
