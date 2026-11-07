"""Проверка перед показом: ролики играют в браузере, файлы на месте, числа сходятся, ничего не устарело."""
import json
import re
import subprocess
from pathlib import Path

import pandas as pd

from mct.summary import check_known_errors
from mct.video import FFMPEG

MAX_CHUNK_MB = 30.0
MAX_SHOWCASE_MB = 60.0
MAX_SHOWCASE_S = 90.0
MAX_REPORT_MB = 30.0
CHUNK_JOIN_S = 0.2
MTIME_TOLERANCE_S = 2.0


def moov_before_mdat(path: Path) -> bool:
    """Служебный блок в начале файла: браузер начинает показ и перематывает, не скачивая файл целиком."""
    with path.open("rb") as file:
        while True:
            header = file.read(8)
            if len(header) < 8:
                return False
            size, kind = int.from_bytes(header[:4], "big"), header[4:]
            if kind == b"moov":
                return True
            if kind == b"mdat" or size == 0:
                return False
            if size == 1:
                size = int.from_bytes(file.read(8), "big") - 8
            file.seek(size - 8, 1)


def _probe(path: Path) -> str:
    return subprocess.run([FFMPEG, "-hide_banner", "-i", str(path)], capture_output=True, text=True).stderr


def video_seconds(path: Path) -> float:
    found = re.search(r"Duration: (\d+):(\d+):([\d.]+)", _probe(path))
    return int(found[1]) * 3600 + int(found[2]) * 60 + float(found[3]) if found else 0.0


def video_problems(path: Path, max_mb: float) -> list[str]:
    if not path.exists():
        return [f"{path}: нет файла"]
    problems = []
    if not re.search(r"Video: h264\b.*yuv420p", _probe(path)):
        problems.append(f"{path.name}: нужны h264 и yuv420p")
    if not moov_before_mdat(path):
        problems.append(f"{path.name}: служебный блок не в начале файла")
    if path.stat().st_size / 1e6 > max_mb:
        problems.append(f"{path.name}: больше {max_mb:.0f} МБ")
    return problems


def _count_problems(tracks_doc: dict, manifest: dict, tracks_csv: Path, results_csv: Path) -> list[str]:
    counts = {
        "total_tracks в tracks.json": tracks_doc["total_tracks"],
        "треков в tracks.json": len(tracks_doc["tracks"]),
        "total_tracks в манифесте": manifest["total_tracks"],
        "строк в tracks.csv": len(pd.read_csv(tracks_csv)),
    }
    problems = [] if len(set(counts.values())) == 1 else [f"Число треков расходится: {counts}"]
    if tracks_csv.read_bytes() != results_csv.read_bytes():
        problems.append(f"{results_csv} отличается от {tracks_csv}")
    return problems


def _chunk_problems(manifest: dict) -> list[str]:
    chunks = manifest["chunks"]
    problems = []
    if not chunks:
        return ["В манифесте нет частей записи"]
    if abs(chunks[0]["t_start_s"] - manifest["clock_start_s"]) > CHUNK_JOIN_S or abs(chunks[-1]["t_end_s"] - manifest["clock_end_s"]) > CHUNK_JOIN_S:
        problems.append("Части не покрывают запись от начала до конца")
    problems += [f"Разрыв между {left['file']} и {right['file']}" for left, right in zip(chunks, chunks[1:])
                 if abs(left["t_end_s"] - right["t_start_s"]) > CHUNK_JOIN_S]
    return problems


def _hint_problems(tracks: list[dict]) -> list[str]:
    by_id = {track["track_id"]: track for track in tracks}
    problems = []
    for track in tracks:
        for hint in track["possibly_same_person_as"]:
            old = by_id.get(hint["track_id"])
            if old is None:
                problems.append(f"#{track['track_id']}: подсказка на несуществующий #{hint['track_id']}")
            elif old["t_end_ms"] > track["t_start_ms"]:
                problems.append(f"#{track['track_id']}: подсказка на #{old['track_id']}, который ещё не закончился")
    return problems


def _stale_problems(source: Path, derived: list[Path]) -> list[str]:
    return [f"{path.name} старше, чем {source.name}: пересоберите" for path in derived if path.stat().st_mtime < source.stat().st_mtime - MTIME_TOLERANCE_S]


def run_checks(tracks_dir: Path, render_dir: Path, results_csv: Path) -> dict[str, list[str]]:
    """Название проверки и найденные проблемы; пустой список значит, что проверка пройдена."""
    tracks_json, tracks_csv = tracks_dir / "tracks.json", tracks_dir / "tracks.csv"
    robustness, crops_index = tracks_dir / "robustness.csv", tracks_dir / "crops" / "index.json"
    hints_sheet = tracks_dir / "gallery" / "hints.jpg"
    manifest_path, showcase = render_dir / "manifest.json", render_dir / "showcase.mp4"
    report, plan = render_dir / "report.html", render_dir / "plan.jpg"
    required = [tracks_json, tracks_csv, robustness, crops_index, hints_sheet, manifest_path, showcase, report, plan, results_csv]
    missing = [f"{path}: нет файла" for path in required if not path.exists()]
    if missing:
        return {"Файлы на месте": missing}

    tracks_doc = json.loads(tracks_json.read_text())
    manifest = json.loads(manifest_path.read_text())
    tracks = tracks_doc["tracks"]
    crops = json.loads(crops_index.read_text())
    videos = [problem for chunk in manifest["chunks"] for problem in video_problems(render_dir / chunk["file"], MAX_CHUNK_MB)]
    videos += video_problems(showcase, MAX_SHOWCASE_MB)
    length = video_seconds(showcase)
    return {
        "Файлы на месте": [],
        "Число треков сходится": _count_problems(tracks_doc, manifest, tracks_csv, results_csv),
        "Ролики играют в браузере": videos,
        "Запись покрыта без разрывов": _chunk_problems(manifest),
        f"Витрина до {MAX_SHOWCASE_S:.0f} с": [] if 0 < length <= MAX_SHOWCASE_S else [f"Длина витрины {length:.1f} с"],
        f"Отчёт до {MAX_REPORT_MB:.0f} МБ": [] if report.stat().st_size / 1e6 <= MAX_REPORT_MB else ["Отчёт слишком большой"],
        "Подсказки ссылаются на прошлые треки": _hint_problems(tracks),
        "Снимки для досье на месте": [f"{item['file']}: нет файла" for item in crops.values() if not (tracks_dir / "crops" / item["file"]).exists()],
        "Примеры ошибок ещё верны": check_known_errors(tracks),
        "Ничего не устарело": _stale_problems(tracks_json, [manifest_path, crops_index, hints_sheet, showcase, report]) + _stale_problems(robustness, [report]),
    }
