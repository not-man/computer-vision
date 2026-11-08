"""Итоги проверок для интерфейса и отчёта: artifacts/test-results/checks.json.

Просмотр склеек и закрытий и таблица людей заполнены глазами для нынешних треков. Если треки изменились, скрипт
останавливается: листы надо нарисовать заново (scripts/draw_audit.py transitions --all, closures --all) и пересмотреть.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

import evaluation.fragment_truth as truth
from mct.demo_check import source_stamp
from scripts.eval_holdout_points import evaluate, read_detections, read_points
from scripts.eval_people import rule_count
from scripts.port_labels import read_tables, transfer_truth_rows

WINDOW_GAP_S = 60.0  # точки дальше друг от друга по времени относятся к разным окнам
PEOPLE_REVIEWER = "Claude по снимкам из исходных кадров, человеком не подтверждено"
HOLDOUT_REVIEWER = "Вторая сессия ИИ по исходным кадрам, без рамок и номеров алгоритма"


def stale_rows(name: str, reviewed: set, current: set) -> str | None:
    """Просмотр сделан для других треков: часть строк устарела или не просмотрена."""
    if reviewed == current:
        return None
    return f"{name}: {len(reviewed - current)} просмотренных строк устарели, {len(current - reviewed)} не просмотрены"


def windows(times: pd.Series) -> list[list[float]]:
    ordered = sorted(float(t) for t in times.unique())
    found = [[ordered[0], ordered[0]]]
    for t in ordered[1:]:
        if t - found[-1][1] > WINDOW_GAP_S:
            found.append([t, t])
        else:
            found[-1][1] = t
    return found


def verdicts(table: pd.DataFrame) -> dict[str, int]:
    return {str(verdict): int(count) for verdict, count in table.verdict.value_counts().items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--perception", type=Path, default=Path("artifacts/perception"))
    parser.add_argument("--tracks", type=Path, default=Path("artifacts/tracks"))
    parser.add_argument("--results", type=Path, default=Path("artifacts/test-results"))
    parser.add_argument("--people", type=Path, default=Path("evaluation/people_truth.csv"))
    args = parser.parse_args()

    tracks = json.loads((args.tracks / "tracks.json").read_text(encoding="utf-8"))["tracks"]
    by_id = {track["track_id"]: track for track in tracks}
    owner = {piece: track["track_id"] for track in tracks for piece in track["tracklets"]}
    transitions = pd.read_csv(args.results / "transition_audit.csv")
    closures = pd.read_csv(args.results / "closure_audit.csv")
    problems = [problem for problem in (
        stale_rows("Склейки", set(zip(transitions.a, transitions.b, transitions.tier)),
                   {(link["a"], link["b"], link["tier"]) for track in tracks for link in track["links"]}),
        stale_rows("Закрытия", set(zip(closures.track, closures.close)), {(track["track_id"], track["close_zone"]) for track in tracks}),
    ) if problem]
    if problems:
        raise SystemExit("Треки изменились после просмотра. " + "; ".join(problems))
    people = pd.read_csv(args.people)
    try:
        extra, _ = rule_count(by_id, people)
    except ValueError as problem:
        raise SystemExit(str(problem))

    observations = pd.read_parquet(args.tracks / "observations.parquet", columns=["camera", "frame", "raw_track", "tracklet"])
    points = read_points()
    holdout, _ = evaluate(points, read_detections(args.perception, sorted(points.camera.unique())), observations, owner)
    marked = transfer_truth_rows(read_tables(args.perception, args.perception))
    crowd = truth.score_fixed_truth(marked.person, truth.track_on_truth(marked, observations, observations.tracklet.map(owner), target="raw_track"))

    wrong = closures[closures.verdict == "неверно"]
    unclear = closures[closures.verdict == "не разобрать"]
    checks = {
        "tracks_json": source_stamp(args.tracks)["tracks.json"],
        "transitions": {"total": len(transitions), "verdicts": verdicts(transitions), "reviewer": " / ".join(transitions.reviewer.unique())},
        "closures": {"total": len(closures), "verdicts": verdicts(closures), "wrong": wrong.track.tolist(), "unclear": unclear.track.tolist(),
                     "reviewer": " / ".join(closures.reviewer.unique())},
        "holdout": {"points": holdout["размечено"], "right": holdout["совпало"], "purity": holdout["чистота"],
                    "windows": windows(points.clock_s), "reviewer": HOLDOUT_REVIEWER},
        "people": {"tracks": len(tracks), "people": int(people.person.nunique()), "by_rule": len(tracks) - len(extra),
                   "extra": [after for _, _, after, _ in extra], "reviewer": PEOPLE_REVIEWER},
        "crowd": {"window": list(truth.WINDOW), "labeled": crowd["размечено"], "matched": crowd["совпало"],
                  "match": crowd["соответствие"], "purity": crowd["чистота"]},
    }
    out = args.results / "checks.json"
    out.write_text(json.dumps(checks, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=False))


if __name__ == "__main__":
    main()
