import json
import os
from pathlib import Path

import numpy as np
import pytest

from mct.demo_check import moov_before_mdat, run_checks, video_problems
from mct.summary import KNOWN_ERRORS
from mct.video import H264Writer


def write_video(path: Path, frames: int = 5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with H264Writer(path, (64, 48), 10.0) as writer:
        for _ in range(frames):
            writer.write(np.zeros((48, 64, 3), np.uint8))


def atom(kind: bytes, payload: bytes = b"xxxx") -> bytes:
    return (len(payload) + 8).to_bytes(4, "big") + kind + payload


@pytest.fixture
def kit(tmp_path: Path) -> dict[str, Path]:
    """Минимальный набор файлов, который проходит все проверки."""
    tracks_dir, render_dir, results = tmp_path / "tracks", tmp_path / "render", tmp_path / "results.csv"
    (tracks_dir / "crops").mkdir(parents=True)
    (tracks_dir / "gallery").mkdir()
    render_dir.mkdir()
    tracks = [{
        "track_id": error.track_id, "t_start_ms": int(error.t_s * 1000), "t_end_ms": int(error.t_s * 1000) + 1000, "start": "", "end": "",
        "possibly_same_person_as": [],
    } for error in KNOWN_ERRORS]
    csv = "track_id,start,end,cameras\n" + "".join(f"{track['track_id']},a,b,c\n" for track in tracks)
    (tracks_dir / "tracks.csv").write_text(csv)
    results.write_text(csv)
    (tracks_dir / "tracks.json").write_text(json.dumps({"total_tracks": len(tracks), "tracks": tracks}))
    (tracks_dir / "robustness.csv").write_text("name\n")
    (tracks_dir / "crops" / "a.jpg").write_bytes(b"x")
    (tracks_dir / "crops" / "index.json").write_text(json.dumps({"1": {"file": "a.jpg"}}))
    (tracks_dir / "gallery" / "hints.jpg").write_bytes(b"x")
    write_video(render_dir / "chunks" / "000.mp4")
    write_video(render_dir / "chunks" / "001.mp4")
    write_video(render_dir / "showcase.mp4")
    (render_dir / "manifest.json").write_text(json.dumps({
        "total_tracks": len(tracks), "clock_start_s": 0.0, "clock_end_s": 120.0,
        "chunks": [{"file": "chunks/000.mp4", "t_start_s": 0.0, "t_end_s": 60.0}, {"file": "chunks/001.mp4", "t_start_s": 60.0, "t_end_s": 120.0}],
    }))
    (render_dir / "report.html").write_text("<html></html>")
    (render_dir / "plan.jpg").write_bytes(b"x")
    return {"tracks_dir": tracks_dir, "render_dir": render_dir, "results_csv": results}


def failed(results: dict[str, list[str]]) -> list[str]:
    return [name for name, problems in results.items() if problems]


def test_good_kit_passes(kit: dict[str, Path]) -> None:
    assert failed(run_checks(**kit)) == []


def test_missing_file_stops_the_rest(kit: dict[str, Path]) -> None:
    (kit["render_dir"] / "showcase.mp4").unlink()
    assert failed(run_checks(**kit)) == ["Файлы на месте"]


def test_count_mismatch_is_caught(kit: dict[str, Path]) -> None:
    path = kit["render_dir"] / "manifest.json"
    manifest = json.loads(path.read_text())
    path.write_text(json.dumps({**manifest, "total_tracks": 999}))
    assert failed(run_checks(**kit)) == ["Число треков сходится"]


def test_stale_list_in_the_repository_is_caught(kit: dict[str, Path]) -> None:
    kit["results_csv"].write_text("track_id,start,end,cameras\n1,a,b,c\n")
    assert failed(run_checks(**kit)) == ["Число треков сходится"]


def test_gap_between_chunks_is_caught(kit: dict[str, Path]) -> None:
    path = kit["render_dir"] / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["chunks"][1]["t_start_s"] = 65.0
    path.write_text(json.dumps(manifest))
    assert failed(run_checks(**kit)) == ["Запись покрыта без разрывов"]


def test_old_render_after_a_new_linking_is_caught(kit: dict[str, Path]) -> None:
    old = (kit["tracks_dir"] / "tracks.json").stat().st_mtime - 100
    os.utime(kit["render_dir"] / "showcase.mp4", (old, old))
    assert failed(run_checks(**kit)) == ["Ничего не устарело"]


def test_hint_to_a_track_that_has_not_ended_is_caught(kit: dict[str, Path]) -> None:
    path = kit["tracks_dir"] / "tracks.json"
    doc = json.loads(path.read_text())
    first, second = doc["tracks"][0], doc["tracks"][1]
    later, earlier = (first, second) if first["t_start_ms"] > second["t_start_ms"] else (second, first)
    earlier["possibly_same_person_as"] = [{"track_id": later["track_id"]}]
    path.write_text(json.dumps(doc))
    assert failed(run_checks(**kit)) == ["Подсказки ссылаются на прошлые треки"]


def test_known_error_that_moved_is_caught(kit: dict[str, Path]) -> None:
    path = kit["tracks_dir"] / "tracks.json"
    doc = json.loads(path.read_text())
    doc["tracks"][0]["t_start_ms"], doc["tracks"][0]["t_end_ms"] = 0, 1000
    path.write_text(json.dumps(doc))
    assert failed(run_checks(**kit)) == ["Примеры ошибок ещё верны"]


def test_moov_position_is_read_from_the_file_structure(tmp_path: Path) -> None:
    good, bad = tmp_path / "good.mp4", tmp_path / "bad.mp4"
    good.write_bytes(atom(b"ftyp") + atom(b"moov") + atom(b"mdat"))
    bad.write_bytes(atom(b"ftyp") + atom(b"mdat") + atom(b"moov"))
    assert moov_before_mdat(good)
    assert not moov_before_mdat(bad)


def test_other_codec_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "not_video.mp4"
    path.write_bytes(atom(b"ftyp") + atom(b"moov"))
    assert any("h264" in problem for problem in video_problems(path, 30.0))
