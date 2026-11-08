"""Витрина: ролик до 90 с из моментов полной записи. Моменты выбираются правилами, поэтому сборка повторяется."""
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterator

import numpy as np

from mct.render import BACKGROUND, CANVAS, FPS, MUTED, TEXT, Stamps, paste
from mct.summary import KNOWN_ERRORS
from mct.video import H264Writer, clock_text, read_frames

BAR_HEIGHT = 64
PAD_S = 1.0
MAX_WALK_S = 26.0
MIN_PAIR_CAMERAS = 3
MIN_PAIR_SHARE = 0.7
MAX_PAIR_DISTANCE_M = 2.0
CROWD_WINDOW_S = 15.0
CROWD_STEP_S = 5.0
LONG_ABSENCE_S = 60.0  # организаторы: отсутствие строго дольше минуты даёт новый номер
RETURN_CLIP_S = 4.0
SHORT_RETURN_S = (5.0, 30.0)  # пауза, которую зритель успеет заметить
CARD_S = 1.5


@dataclass(frozen=True)
class Moment:
    title: str
    note: str
    spans: tuple[tuple[float, float], ...]
    card: str = ""


def reel_seconds(moments: list[Moment]) -> float:
    return sum(end - start for moment in moments for start, end in moment.spans) + sum(CARD_S for moment in moments if moment.card)


def _span(track: dict) -> tuple[float, float]:
    return track["t_start_ms"] / 1000, track["t_end_ms"] / 1000


def _duration(track: dict) -> float:
    start, end = _span(track)
    return end - start


def _clean(track: dict) -> bool:
    """Показываем как пример только то, в чём уверены: без спорных связей и не трек из известных ошибок."""
    return track["track_id"] not in {error.track_id for error in KNOWN_ERRORS} and all(link["confidence"] != "yellow" for link in track["links"])


def _crowding(track: dict, tracks: list[dict]) -> float:
    start, end = _span(track)
    shared = sum(max(0.0, min(end, _span(other)[1]) - max(start, _span(other)[0])) for other in tracks if other is not track)
    return shared / max(end - start, 0.001)


def _camera_order(track: dict) -> str:
    order: list[str] = []
    for segment in sorted((s for s in track["segments"] if s["kind"] == "observed"), key=lambda s: s["t_start_ms"]):
        camera = segment["camera_id"][-1]
        if not order or order[-1] != camera:
            order.append(camera)
    return " → ".join(order)


def _walk(tracks: list[dict]) -> Moment | None:
    candidates = [track for track in tracks if _clean(track) and _duration(track) <= MAX_WALK_S]
    if not candidates:
        return None
    widest = max(len(track["cameras"]) for track in candidates)
    track = min((t for t in candidates if len(t["cameras"]) == widest), key=lambda t: (_crowding(t, tracks), _duration(t), t["track_id"]))
    start, end = _span(track)
    return Moment("Один номер на всём пути", f"#{track['track_id']} проходит камеры {_camera_order(track)}. Номер и цвет не меняются.",
                  ((start - PAD_S, end + PAD_S),))


def _median_distance(a: dict, b: dict) -> float:
    path_a, path_b = np.array(a["plan_path"], float), np.array(b["plan_path"], float)
    if not len(path_a) or not len(path_b):
        return float("inf")
    shared = path_a[(path_a[:, 2] >= path_b[0, 2]) & (path_a[:, 2] <= path_b[-1, 2])]
    if not len(shared):
        return float("inf")
    return float(np.median(np.hypot(shared[:, 0] - np.interp(shared[:, 2], path_b[:, 2], path_b[:, 0]),
                                    shared[:, 1] - np.interp(shared[:, 2], path_b[:, 2], path_b[:, 1]))))


def _pair(tracks: list[dict]) -> Moment | None:
    best, best_shared = None, 0.0
    walkers = [track for track in tracks if _clean(track) and len(track["cameras"]) >= MIN_PAIR_CAMERAS]
    for a, b in combinations(walkers, 2):
        (start_a, end_a), (start_b, end_b) = _span(a), _span(b)
        whole = max(end_a, end_b) - min(start_a, start_b)
        shared = min(end_a, end_b) - max(start_a, start_b)
        if whole > MAX_WALK_S or shared < MIN_PAIR_SHARE * whole or shared <= best_shared:
            continue
        if len(set(a["cameras"]) & set(b["cameras"])) >= MIN_PAIR_CAMERAS and _median_distance(a, b) <= MAX_PAIR_DISTANCE_M:
            best, best_shared = (a, b), shared
    if best is None:
        return None
    a, b = best
    start, end = min(_span(a)[0], _span(b)[0]), max(_span(a)[1], _span(b)[1])
    return Moment("Двое рядом: у каждого свой номер", f"#{a['track_id']} и #{b['track_id']} идут вместе через несколько камер, номера не смешиваются.",
                  ((start - PAD_S, end + PAD_S),))


def _long_return(tracks: list[dict]) -> Moment | None:
    by_id = {track["track_id"]: track for track in tracks}
    hints = [(hint, track) for track in tracks for hint in track["possibly_same_person_as"]
             if hint["absent_s"] >= LONG_ABSENCE_S and hint["track_id"] in by_id]
    if not hints:
        return None
    hint, new = max(hints, key=lambda item: (item[0]["r"], item[1]["track_id"]))
    old = by_id[hint["track_id"]]
    left, back = _span(old)[1], _span(new)[0]
    absent = int(round(hint["absent_s"]))
    note = (f"#{old['track_id']} ушёл в {clock_text(left)[:8]}, #{new['track_id']} появился в {clock_text(back)[:8]}. "
            f"По ТЗ это новый трек; сходство {hint['r']:.2f} остаётся подсказкой, номера не склеены.")
    return Moment("Ушёл надолго и вернулся: новый номер и подсказка", note,
                  ((left - RETURN_CLIP_S, left + PAD_S), (back - PAD_S, back + RETURN_CLIP_S)), f"прошло {absent // 60} мин {absent % 60:02d} с")


def _short_return(tracks: list[dict]) -> Moment | None:
    """Вернулся быстрее минуты: номер тот же. Берём самое уверенное совпадение с заметной паузой."""
    found = [(link["r"], track, link) for track in tracks for link in track["links"]
             if link.get("tier") == "return" and SHORT_RETURN_S[0] <= -link["contact_s"] <= SHORT_RETURN_S[1]]
    if not found:
        return None
    r, track, link = max(found, key=lambda item: (item[0], item[1]["track_id"]))
    left, pause = link["t_ms"] / 1000, -link["contact_s"]
    return Moment("Вернулся быстрее минуты: номер тот же", f"#{track['track_id']} нет ни на одной камере {pause:.0f} с. Организаторы считают длительным отсутствие дольше минуты; сходство {r:.2f}.",
                  ((left - RETURN_CLIP_S, left + PAD_S), (left + pause - PAD_S, left + pause + RETURN_CLIP_S)), f"прошло {pause:.0f} с")


def _tracks_in_window(tracks: list[dict], start: float, end: float) -> int:
    return sum(any(segment["kind"] == "observed" and segment["t_start_ms"] / 1000 < end and start < segment["t_end_ms"] / 1000
                   for segment in track["segments"]) for track in tracks)


def _crowd(tracks: list[dict], taken: list[tuple[float, float]]) -> Moment | None:
    if not tracks:
        return None
    first, last = min(_span(t)[0] for t in tracks), max(_span(t)[1] for t in tracks)
    best, best_count = None, 0
    for start in np.arange(first, max(first + CROWD_STEP_S, last - CROWD_WINDOW_S), CROWD_STEP_S):
        end = float(start) + CROWD_WINDOW_S
        if any(start < taken_end and taken_start < end for taken_start, taken_end in taken):
            continue
        count = _tracks_in_window(tracks, float(start), end)
        if count > best_count:
            best, best_count = (float(start), end), count
    if best is None:
        return None
    return Moment("Много людей одновременно", f"Треков на камерах за эти {CROWD_WINDOW_S:.0f} с: {best_count}.", (best,))


def pick_moments(tracks: list[dict]) -> list[Moment]:
    """Порядок показа: простой случай, затем сложнее. Людный момент выбирается последним, чтобы кадры не повторялись."""
    walk, pair, short, back = _walk(tracks), _pair(tracks), _short_return(tracks), _long_return(tracks)
    taken = [span for moment in (walk, pair, short, back) if moment for span in moment.spans]
    return [moment for moment in (walk, pair, _crowd(tracks, taken), short, back) if moment]


def with_bar(frame: np.ndarray, moment: Moment, index: int, total: int, stamps: Stamps) -> np.ndarray:
    canvas = np.full((frame.shape[0] + BAR_HEIGHT, frame.shape[1], 3), BACKGROUND, np.uint8)
    canvas[BAR_HEIGHT:] = frame
    paste(canvas, stamps.text(f"{index} из {total}. {moment.title}", 22, TEXT, bold=True), 16, 7)
    paste(canvas, stamps.text(moment.note, 15, MUTED), 16, 38)
    return canvas


def _span_frames(manifest: dict, render_dir: Path, start_s: float, end_s: float) -> Iterator[np.ndarray]:
    for chunk in manifest["chunks"]:
        first, last = max(start_s, chunk["t_start_s"]), min(end_s, chunk["t_end_s"])
        if first < last:
            for _, frame in read_frames(render_dir / chunk["file"], CANVAS, FPS, FPS, first - chunk["t_start_s"], last - first):
                yield frame


def build_showcase(moments: list[Moment], manifest: dict, render_dir: Path, out: Path) -> float:
    """Пишет ролик и возвращает его длину в секундах."""
    stamps = Stamps()
    frames = 0
    blank = np.full((CANVAS[1], CANVAS[0], 3), BACKGROUND, np.uint8)
    with H264Writer(out, (CANVAS[0], CANVAS[1] + BAR_HEIGHT), FPS) as writer:
        for index, moment in enumerate(moments, 1):
            for number, (start, end) in enumerate(moment.spans):
                if number and moment.card:
                    card = with_bar(blank, moment, index, len(moments), stamps)
                    text = stamps.text(moment.card, 30, TEXT, bold=True)
                    paste(card, text, (CANVAS[0] - text.shape[1]) // 2, BAR_HEIGHT + (CANVAS[1] - text.shape[0]) // 2)
                    for _ in range(int(CARD_S * FPS)):
                        writer.write(card)
                        frames += 1
                for frame in _span_frames(manifest, render_dir, start, end):
                    writer.write(with_bar(frame, moment, index, len(moments), stamps))
                    frames += 1
    return frames / FPS
