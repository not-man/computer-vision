"""Демо сквозного трекинга. Приложение ничего не считает: читает только готовые файлы."""
import json
import shutil
from pathlib import Path

import pandas as pd
import streamlit as st

from mct.charts import TIMELINE_HEIGHT, camera_label, curve_figure, plan_figure, timeline_figure
from mct.render import CLOSE_TEXT
from mct.video import clock_text

ROOT = Path(__file__).resolve().parents[1]
RENDER = ROOT / "render" / "full"
TRACKS = ROOT / "artifacts" / "tracks"
PLAN_IMAGE = RENDER / "plan.jpg"
CHART_CONFIG = {"displayModeBar": False}
MAX_CROPS = 8
TABLE_COLUMNS = {
    "Трек": st.column_config.NumberColumn(width=50),
    "Начало": st.column_config.TextColumn(width=72),
    "Конец": st.column_config.TextColumn(width=72),
    "Секунд": st.column_config.NumberColumn(width=60),
    "Камеры": st.column_config.TextColumn(width=86),
    "Закрыт": st.column_config.TextColumn(width=210),
    "Похож на": st.column_config.TextColumn(width=70),
}

st.set_page_config(layout="wide", page_title="Сквозной трекинг", page_icon="🎥")


@st.cache_data
def load() -> tuple[dict, dict, dict, pd.DataFrame | None]:
    tracks = json.loads((TRACKS / "tracks.json").read_text())
    manifest = json.loads((RENDER / "manifest.json").read_text())
    crops_path = TRACKS / "crops" / "index.json"
    crops = json.loads(crops_path.read_text()) if crops_path.exists() else {}
    robustness = TRACKS / "robustness.csv"
    return tracks, manifest, crops, pd.read_csv(robustness) if robustness.exists() else None


TRACKS_DATA, MANIFEST, CROPS, ROBUSTNESS = load()
ALL_TRACKS = TRACKS_DATA["tracks"]
BY_ID = {track["track_id"]: track for track in ALL_TRACKS}


def duration_s(track: dict) -> float:
    return (track["t_end_ms"] - track["t_start_ms"]) / 1000


def summary_row() -> None:
    hints = sum(len(track["possibly_same_person_as"]) for track in ALL_TRACKS)
    length = MANIFEST["clock_end_s"] - MANIFEST["clock_start_s"]
    columns = st.columns(4)
    columns[0].metric("Треков по ТЗ", TRACKS_DATA["total_tracks"])
    columns[1].metric("Камер", len({camera for track in ALL_TRACKS for camera in track["cameras"]}))
    columns[2].metric("Длина записи", f"{length / 60:.0f} мин")
    columns[3].metric("Подсказок о возврате", hints)


def screen_showcase() -> None:
    summary_row()
    showcase = RENDER / "showcase.mp4"
    if showcase.exists():
        st.video(str(showcase))
        st.caption("Фрагменты полной записи. Вся запись доступна на следующей вкладке.")
    else:
        st.info("Витрина ещё не собрана: scripts/make_showcase.py")


def screen_recording() -> None:
    chunks = MANIFEST["chunks"]
    labels = [chunk["start"][:8] for chunk in chunks]
    label = st.select_slider("Начало минуты", options=labels)
    chunk = chunks[labels.index(label)]
    st.video(str(RENDER / chunk["file"]))
    st.caption("Секунды в углах ячеек могут различаться: это часы камер. Кадры выровнены по общему времени.")
    events = [event for event in MANIFEST["events"] if chunk["t_start_s"] <= event["t_s"] < chunk["t_end_s"]]
    if events:
        st.dataframe(pd.DataFrame({"Время": [clock_text(event["t_s"])[:8] for event in events], "Событие": [event["text"] for event in events]}),
                     hide_index=True, width="stretch", height=240)


def tracks_table() -> pd.DataFrame:
    return pd.DataFrame([{
        "Трек": track["track_id"],
        "Начало": track["start"][:8],
        "Конец": track["end"][:8],
        "Секунд": round(duration_s(track)),
        "Камеры": ", ".join(camera[-1] for camera in track["cameras"]),
        "Закрыт": CLOSE_TEXT.get(track["close_zone"], track["close_zone"]),
        "Похож на": ", ".join(f"#{hint['track_id']}" for hint in track["possibly_same_person_as"]),
    } for track in ALL_TRACKS])


def track_chunks(track: dict) -> list[dict]:
    start, end = track["t_start_ms"] / 1000, track["t_end_ms"] / 1000
    return [chunk for chunk in MANIFEST["chunks"] if chunk["t_start_s"] < end and start < chunk["t_end_s"]]


def show_crops(track: dict) -> None:
    items = [CROPS[str(tracklet)] for tracklet in track["tracklets"] if str(tracklet) in CROPS][:MAX_CROPS]
    if not items:
        return
    columns = st.columns(MAX_CROPS)
    for column, item in zip(columns, items):
        column.image(str(TRACKS / "crops" / item["file"]), caption=f"{camera_label(item['camera'])} {clock_text(item['t_s'])[3:8]}")


def show_links(track: dict) -> None:
    if not track["links"]:
        st.caption("Трек видела одна камера, связей между камерами нет.")
        return
    tiers = {"sustained": "перекрытие", "brief": "короткий контакт", "gap": "разрыв"}
    st.dataframe(pd.DataFrame([{
        "Время": clock_text(link["t_ms"] / 1000)[:8],
        "Откуда": camera_label(link["camera_a"]),
        "Куда": camera_label(link["camera_b"]),
        "Вид": tiers[link["tier"]],
        "G, м": link["g_m"],
        "Контакт, с": link["contact_s"],
        "R": link["r"],
        "Оценка": "спорная" if link["confidence"] == "yellow" else "уверенная",
    } for link in track["links"]]), hide_index=True, width="stretch")


def show_dossier(track: dict) -> None:
    st.subheader(f"Трек #{track['track_id']}")
    columns = st.columns(4)
    columns[0].metric("Начало", track["start"][:8])
    columns[1].metric("Конец", track["end"][:8])
    columns[2].metric("Длительность", f"{duration_s(track):.0f} с")
    columns[3].metric("Камеры", ", ".join(camera[-1] for camera in track["cameras"]))
    reason = CLOSE_TEXT.get(track["close_zone"], track["close_zone"])
    st.write(f"Закрыт: {reason}.")
    for hint in track["possibly_same_person_as"]:
        absent = int(round(hint["absent_s"]))
        st.write(f"Вероятно, вернулся #{hint['track_id']}: отсутствовал {absent // 60}:{absent % 60:02d}, сходство {hint['r']:.2f}. Не склеено по правилу ТЗ.")
    show_crops(track)

    left, right = st.columns([3, 2])
    with left:
        chunks = track_chunks(track)
        chosen = chunks[0]
        if len(chunks) > 1:
            labels = [chunk["start"][:8] for chunk in chunks]
            chosen = chunks[labels.index(st.selectbox("Минута записи", labels, key=f"minute_{track['track_id']}"))]
        start = max(track["t_start_ms"] / 1000 - chosen["t_start_s"] - 1, 0)
        st.video(str(RENDER / chosen["file"]), start_time=int(start))
        show_links(track)
    with right:
        st.plotly_chart(plan_figure(track, PLAN_IMAGE, ROOT / "configs"), config=CHART_CONFIG, width="stretch", key=f"plan_{track['track_id']}")


@st.fragment
def screen_tracks() -> None:
    table = tracks_table()
    left, right = st.columns([5, 6])
    selection = left.dataframe(table, hide_index=True, width="stretch", height=TIMELINE_HEIGHT, column_config=TABLE_COLUMNS,
                               on_select="rerun", selection_mode="single-row", key="track_table")
    rows = selection.selection.rows
    selected = int(table["Трек"].iloc[rows[0]]) if rows else None
    right.plotly_chart(timeline_figure(ALL_TRACKS, selected), config=CHART_CONFIG, width="stretch")
    if selected is None:
        st.info("Выберите строку в таблице, чтобы открыть досье трека.")
    else:
        show_dossier(BY_ID[selected])


def screen_returns() -> None:
    st.write("Слева ушедший трек, справа новый. Подсказка никогда не меняет ID и подсчёт: по ТЗ вернувшийся получает новый номер.")
    sheet = TRACKS / "gallery" / "hints.jpg"
    if sheet.exists():
        st.image(str(sheet))
    else:
        st.info("Лист ещё не собран: scripts/draw_gallery.py")


def screen_honesty() -> None:
    st.write(
        "Эталонной разметки нет, поэтому качество проверено глазами по снимкам, а не измерено. "
        "Совпадение ниже показывает стабильность результата, а не его правильность."
    )
    if ROBUSTNESS is None:
        st.info("Проверка ещё не запускалась: scripts/check_robustness.py")
        return
    scaled = ROBUSTNESS[ROBUSTNESS.group == "пороги"]
    columns = st.columns(3)
    columns[0].metric("Порог ±20%: треков", f"{scaled.tracks.min()}–{scaled.tracks.max()}")
    columns[1].metric("Совпадение наблюдений, минимум", f"{scaled.agreement.min():.3f}")
    columns[2].metric("Вариантов проверено", len(ROBUSTNESS))

    left, right = st.columns(2)
    duration = ROBUSTNESS[ROBUSTNESS.group == "кривая длительности"]
    veto = ROBUSTNESS[ROBUSTNESS.group == "кривая r_veto"]
    left.plotly_chart(curve_figure(duration.name.str.split().str[1].astype(float).reset_index(drop=True), duration.tracks.reset_index(drop=True),
                                   "Число треков зависит от минимальной длительности", "минимальная длительность трека, с", 2.0), config=CHART_CONFIG, width="stretch")
    right.plotly_chart(curve_figure(veto.name.str.split().str[1].astype(float).reset_index(drop=True), veto.tracks.reset_index(drop=True),
                                    "Число треков зависит от запрета по внешности", "порог R_veto", 0.05), config=CHART_CONFIG, width="stretch")

    st.subheader("Что будет, если отключить правило")
    removed = ROBUSTNESS[ROBUSTNESS.group == "откл"].sort_values("agreement")
    st.dataframe(pd.DataFrame({"Правило": removed.name, "Треков": removed.tracks, "Спорных связей": removed.yellow, "Совпадение": removed.agreement}),
                 hide_index=True, width="stretch")

    st.subheader("Известные ошибки")
    st.markdown(
        "- **Подмена вдали и в толпе.** Трекер одной камеры может перепутать двух идущих рядом людей, тогда в треке появляется чужой фрагмент.\n"
        "- **Пары вплотную.** Если двое идут рядом и одеты похоже, передача отмечается как спорная: номера могут поменяться местами, число треков при этом верно.\n"
        "- **Уход и возврат через несколько секунд.** Сейчас это новый трек с подсказкой: ТЗ не называет, что считать длительным отсутствием."
    )


SCREENS = {
    "Витрина": screen_showcase,
    "Вся запись": screen_recording,
    "Треки": screen_tracks,
    "Возвраты": screen_returns,
    "Как проверяли": screen_honesty,
}

if not PLAN_IMAGE.exists():
    shutil.copy(ROOT.parent / "data" / "office.jpg", PLAN_IMAGE)
SCREENS[st.radio("Раздел", list(SCREENS), horizontal=True, label_visibility="collapsed")]()
