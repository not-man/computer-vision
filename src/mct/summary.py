"""Что показывают интерфейс и офлайн-отчёт: строки таблиц и известные ошибки. Один источник, чтобы экраны не расходились."""
from dataclasses import dataclass

import pandas as pd

from mct.charts import camera_label
from mct.render import CLOSE_TEXT
from mct.video import clock_text

LINK_TIERS = {"sustained": "перекрытие", "brief": "короткий контакт", "gap": "разрыв", "return": "возврат до минуты"}
SEGMENT_JOIN_S = 1.0  # соседние куски одной камеры с таким разрывом показываются одной строкой
HONESTY_NOTE = ("Полной эталонной разметки нет. Числа ниже измерены на толпе в 22 секунды, размеченной вручную по одежде, и на парах треков "
                "из просмотра всей записи глазами. По этим примерам настраивали правила, поэтому оценка скорее оптимистичная. "
                "Независимая проверка ниже: точки в окнах, где правила не настраивались, и просмотр всех склеек и закрытий. "
                "Совпадение при изменении порогов показывает стабильность результата, а не его правильность.")
CHECKS_NOTE = ("Каждую склейку и каждое закрытие смотрели по снимкам из исходных кадров. Это сделал ИИ, человеком пока не подтверждено: "
               "это проверка, а не эталон.")
EXPERIMENTS_NOTE = ("В таблице — прежние опыты относительно базовой версии с 61 треком. У другой модели внешности пороги приведены к той же доле пар. "
                    "Толпа: для прогонов детектора и порога разреза доля на всех 1639 размеченных рамках (база 0.922); "
                    "для остальных — по наблюдениям, вошедшим в треки (база 0.926). Текущий результат показан выше отдельно.")
CLEAR_VERDICTS = {"один человек", "скорее один", "на известной ошибке", "не разобрать"}


@dataclass(frozen=True)
class KnownError:
    title: str
    text: str
    track_id: int
    t_s: float  # момент примера, секунды от 17:00:00; demo_check проверяет, что после пересчёта пример ещё верен


KNOWN_ERRORS = [
    KnownError("Перескок в плотной толпе",
               "17:45:33–17:45:39, cam_3: #32 начинается на человеке в чёрной толстовке, а продолжается на человеке в куртке. "
               "Трекер дальней камеры перескочил, а доказательств с двух сторон для разреза не хватило.", 32, 2735.0),
    KnownError("Двое в обнимку",
               "17:43:46–17:44:03: один человек несёт другого, детектор даёт одну маску, и у обоих один номер #21. "
               "Пока они не разойдутся, второго человека система не видит.", 21, 2628.0),
    KnownError("Лишний короткий трек",
               "17:41:16–17:41:22, cam_3: человек в чёрной футболке идёт вплотную к человеку в белом, и в рамку попадает сосед. "
               "Сходство с его же треком #11 падает почти до нуля, поэтому он получает отдельный #13. Счёт завышен на единицу.", 13, 2480.0),
]


def duration_s(track: dict) -> float:
    return (track["t_end_ms"] - track["t_start_ms"]) / 1000


def close_text(track: dict) -> str:
    return CLOSE_TEXT.get(track["close_zone"], track["close_zone"])


def camera_numbers(track: dict) -> str:
    return ", ".join(camera[-1] for camera in track["cameras"])


def track_rows(tracks: list[dict]) -> list[dict]:
    return [{
        "Трек": track["track_id"],
        "Начало": track["start"][:8],
        "Конец": track["end"][:8],
        "Секунд": round(duration_s(track)),
        "Камеры": camera_numbers(track),
        "Закрыт": close_text(track),
        "Похож на": ", ".join(f"#{hint['track_id']}" for hint in track["possibly_same_person_as"]),
    } for track in tracks]


def segment_rows(track: dict) -> list[dict]:
    """На каких камерах и когда проходил трек."""
    pieces: dict[str, list[list[int]]] = {}
    for segment in sorted((s for s in track["segments"] if s["kind"] == "observed"), key=lambda s: s["t_start_ms"]):
        camera = pieces.setdefault(segment["camera_id"], [])
        if camera and segment["t_start_ms"] - camera[-1][1] <= SEGMENT_JOIN_S * 1000:
            camera[-1][1] = max(camera[-1][1], segment["t_end_ms"])
        else:
            camera.append([segment["t_start_ms"], segment["t_end_ms"]])
    rows = [(start, end, camera) for camera, spans in pieces.items() for start, end in spans]
    return [{"Камера": camera_label(camera), "С": clock_text(start / 1000)[:8], "По": clock_text(end / 1000)[:8], "Секунд": round((end - start) / 1000)}
            for start, end, camera in sorted(rows)]


def track_video_chunks(track: dict, chunks: list[dict]) -> list[dict]:
    """Минуты, где трек виден, и момент первого наблюдения в каждой из них."""
    observed = [segment for segment in track["segments"] if segment["kind"] == "observed"]
    result = []
    for chunk in chunks:
        start, end = chunk["t_start_s"], chunk["t_end_s"]
        visible = [segment for segment in observed
                   if segment["t_start_ms"] / 1000 < end and start < segment["t_end_ms"] / 1000]
        if not visible:
            continue
        first = min(max(start, segment["t_start_ms"] / 1000) for segment in visible)
        cameras = sorted({segment["camera_id"] for segment in visible})
        result.append({
            "label": chunk["start"][:8], "file": chunk["file"], "start_s": start,
            "offset": round(max(first - start - 1, 0), 1),
            "first_seen": clock_text(first)[:8],
            "cameras": ", ".join(camera_label(camera) for camera in cameras),
        })
    return result


def link_rows(track: dict) -> list[dict]:
    return [{
        "Время": clock_text(link["t_ms"] / 1000)[:8],
        "Откуда": camera_label(link["camera_a"]),
        "Куда": camera_label(link["camera_b"]),
        "Вид": LINK_TIERS[link["tier"]],
        "G, м": link["g_m"],
        "Контакт, с": link["contact_s"],
        "R": link["r"],
        "Оценка": "спорная" if link["confidence"] == "yellow" else "уверенная",
    } for link in track["links"]]


def hint_texts(track: dict) -> list[str]:
    texts = []
    for hint in track["possibly_same_person_as"]:
        absent = int(round(hint["absent_s"]))
        texts.append(f"Вероятно, вернулся #{hint['track_id']}: отсутствовал {absent // 60}:{absent % 60:02d}, сходство {hint['r']:.2f}. Не склеено по правилу ТЗ.")
    return texts


def rule_rows(robustness: pd.DataFrame) -> list[dict]:
    removed = robustness[robustness.group == "откл"].sort_values("agreement")
    return [{"Правило": row.name, "Треков": int(row.tracks), "Спорных связей": int(row.yellow), "Совпадение": round(float(row.agreement), 3)}
            for row in removed.itertuples()]


def quality_tiles(tracks_doc: dict, checks: dict | None = None) -> list[tuple[str, str]]:
    """Измеренное качество из tracks.json. Если его нет, показывать нечего.

    Соответствие на толпе берётся из итогов проверок, если они есть: там знаменатель все размеченные рамки, а не только вошедшие в треки.
    """
    quality = tracks_doc.get("quality")
    if not quality:
        return []
    crowd = f"{checks['crowd']['match']:.3f}" if checks else f"{quality['crowd']['match']:.2f}"
    tiles = [("Толпа: соответствие человек-трек", crowd),
             ("Толпа: людей и треков", f"{quality['crowd']['people']} и {quality['crowd']['tracks']}"),
             ("Пары, проверенные глазами: верно", f"{quality['pairs']['right']} из {quality['pairs']['total']}")]
    if "tracks_without_returns" in quality:
        tiles.append(("Треков без возвратов по внешности", str(quality["tracks_without_returns"])))
    return tiles


def _clock_range(window: list[float]) -> str:
    start, end = clock_text(window[0])[:8], clock_text(window[1])[:8]
    return f"{start}–{end[6:]}" if start[:5] == end[:5] else f"{start}–{end}"


def _tracks_text(tracks: list[int]) -> str:
    return ", ".join(f"#{track}" for track in tracks)


def check_rows(checks: dict) -> list[dict]:
    """Итоги проверок по всей записи из artifacts/test-results/checks.json: что смотрели, что получилось, кто смотрел."""
    transitions, closures, holdout, people, crowd = (checks[key] for key in ("transitions", "closures", "holdout", "people", "crowd"))
    linked = ", ".join(f"{verdict} {count}" for verdict, count in transitions["verdicts"].items())
    if set(transitions["verdicts"]) <= CLEAR_VERDICTS:
        linked += "; явно разных людей нет"
    named = {"неверно": closures["wrong"], "не разобрать": closures["unclear"]}
    closed = ", ".join(f"{verdict} {count}" + (f" ({_tracks_text(named[verdict])})" if named.get(verdict) else "")
                       for verdict, count in closures["verdicts"].items())
    points = f"на своём треке {holdout['right']} из {holdout['points']}"
    points += ", смешений нет" if holdout["purity"] == 1.0 else f", чистота {holdout['purity']:.3f}"
    return [
        {"Проверка": f"Все склейки кусков: {transitions['total']}", "Результат": linked, "Кто смотрел": transitions["reviewer"]},
        {"Проверка": f"Все закрытия треков: {closures['total']}", "Результат": closed, "Кто смотрел": closures["reviewer"]},
        {"Проверка": f"Независимые точки: {holdout['points']} в окнах {', '.join(_clock_range(window) for window in holdout['windows'])}",
         "Результат": points, "Кто смотрел": f"{holdout['reviewer']}; правила по этим окнам не настраивались"},
        {"Проверка": f"Треки по людям на снимках: {people['tracks']}",
         "Результат": f"{people['people']} человек; по правилу минуты треков должно быть {people['by_rule']}, лишние {_tracks_text(people['extra'])}",
         "Кто смотрел": people["reviewer"]},
        {"Проверка": f"Толпа {_clock_range(crowd['window'])}: {crowd['labeled']} размеченных рамок",
         "Результат": f"на своём треке {crowd['matched']} ({crowd['match']:.3f}), чистота {crowd['purity']:.3f}",
         "Кто смотрел": "Разметка вручную по одежде; по ней настраивали правила"},
    ]


def robustness_tiles(robustness: pd.DataFrame) -> list[tuple[str, str]]:
    scaled = robustness[robustness.group == "пороги"]
    return [("Порог ±20%: треков", f"{scaled.tracks.min()}–{scaled.tracks.max()}"),
            ("Совпадение наблюдений, минимум", f"{scaled.agreement.min():.3f}"),
            ("Вариантов проверено", str(len(robustness)))]


def check_known_errors(tracks: list[dict]) -> list[str]:
    """После пересчёта номера могут сдвинуться: тогда примеры в интерфейсе надо переписать."""
    by_id = {track["track_id"]: track for track in tracks}
    problems = []
    for error in KNOWN_ERRORS:
        track = by_id.get(error.track_id)
        if track is None:
            problems.append(f"«{error.title}»: нет трека #{error.track_id}")
        elif not track["t_start_ms"] / 1000 - 1 <= error.t_s <= track["t_end_ms"] / 1000 + 1:
            problems.append(f"«{error.title}»: трек #{error.track_id} идёт {track['start'][:8]}–{track['end'][:8]}, а пример в {clock_text(error.t_s)[:8]}")
    return problems
