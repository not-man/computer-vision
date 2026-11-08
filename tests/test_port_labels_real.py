"""Проверка переноса разметки на настоящих сохранённых детекциях проекта."""
import json
from pathlib import Path

import pandas as pd

import evaluation.fragment_truth as truth
from mct.perception import valid_embeddings, valid_perception_output
from scripts.port_labels import port_labels, read_tables, transfer_truth_rows


ROOT = Path(__file__).resolve().parents[1]
PERCEPTION = ROOT / "artifacts" / "perception"


def test_identity_transfer_preserves_every_labeled_observation() -> None:
    tables = read_tables(PERCEPTION, PERCEPTION)
    marked = transfer_truth_rows(tables)
    expected = sum(int(truth.label_rows(old).notna().sum()) for old, _ in tables.values())
    assert len(marked) == expected == 1639
    assert marked.new_raw_track.notna().all()
    assert (marked.new_raw_track.astype(int) == marked.raw_track).all()
    assert (marked.match_iou == 1.0).all()


def test_clocked_source_agrees_with_saved_baseline_observations() -> None:
    tables = read_tables(PERCEPTION, PERCEPTION)
    assigned = pd.read_parquet(ROOT / "artifacts" / "tracks" / "observations.parquet",
                               columns=["camera", "frame", "raw_track", "t_s"])
    for camera, (old, _) in tables.items():
        matched = old[["frame", "raw_track", "t_s"]].merge(
            assigned.loc[assigned.camera == camera, ["frame", "raw_track", "t_s"]],
            on=["frame", "raw_track"], suffixes=("_clocked", "_saved"), validate="one_to_one")
        assert not matched.empty
        assert (matched.t_s_clocked - matched.t_s_saved).abs().max() < 1e-9


def test_interval_migration_preserves_real_baseline_labels(monkeypatch) -> None:
    tables = read_tables(PERCEPTION, PERCEPTION)
    before = {camera: truth.label_rows(old) for camera, (old, _) in tables.items()}
    intervals, total, kept = port_labels(tables)
    assert total == kept == 1639
    monkeypatch.setattr(truth, "LABELS", intervals)
    for camera, (old, _) in tables.items():
        assert truth.label_rows(old).fillna("-").equals(before[camera].fillna("-"))


def test_fixed_score_reproduces_saved_result() -> None:
    tables = read_tables(PERCEPTION, PERCEPTION)
    marked = transfer_truth_rows(tables)
    folder = ROOT / "artifacts" / "tracks"
    assigned = pd.read_parquet(folder / "observations.parquet",
                               columns=["camera", "frame", "raw_track", "tracklet"])
    tracks = json.loads((folder / "tracks.json").read_text(encoding="utf-8"))["tracks"]
    owner = {piece: track["track_id"] for track in tracks for piece in track["tracklets"]}
    prediction = truth.track_on_truth(marked, assigned, assigned.tracklet.map(owner), target="raw_track")
    result = truth.score_fixed_truth(marked.person, prediction)
    assert result == {"размечено": 1639, "в_треках": 1624, "совпало": 1551,
                      "соответствие": 0.946, "чистота": 0.955}


def test_saved_tracks_restart_after_more_than_one_minute_without_observation() -> None:
    tracks = json.loads((ROOT / "artifacts" / "tracks" / "tracks.json").read_text(encoding="utf-8"))["tracks"]
    for track in tracks:
        observed = sorted((segment for segment in track["segments"] if segment["kind"] == "observed"),
                          key=lambda segment: segment["t_start_ms"])
        assert observed, f"#{track['track_id']}: нет наблюдений"
        covered_until = observed[0]["t_end_ms"]
        for segment in observed[1:]:
            assert segment["t_start_ms"] - covered_until <= 60_000, (
                f"#{track['track_id']}: ID сохранён после отсутствия дольше минуты")
            covered_until = max(covered_until, segment["t_end_ms"])


def test_incomplete_real_embedding_is_not_treated_as_finished(tmp_path: Path) -> None:
    source = PERCEPTION / "camera_new_1_emb.npz"
    assert valid_perception_output(PERCEPTION, "camera_new_1")
    assert not valid_perception_output(PERCEPTION, "camera_new_1", {"imgsz": -1})
    damaged = tmp_path / source.name
    with source.open("rb") as stream:
        damaged.write_bytes(stream.read(100))
    assert not valid_embeddings(damaged)
