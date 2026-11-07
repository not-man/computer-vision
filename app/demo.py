"""Демо сквозного трекинга. Приложение ничего не считает: читает только готовые файлы."""
import json
import shutil
from pathlib import Path

import pandas as pd
import streamlit as st

from mct.charts import TIMELINE_HEIGHT, camera_label, plan_figure, robustness_curves, timeline_figure
from mct.summary import (
    HONESTY_NOTE, KNOWN_ERRORS, camera_numbers, close_text, duration_s, hint_texts, link_rows, quality_tiles, robustness_tiles, rule_rows, segment_rows, track_rows,
)
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
    tracks = json.loads((TRACKS / "tracks.json").read_text(encoding="utf-8"))
    manifest = json.loads((RENDER / "manifest.json").read_text(encoding="utf-8"))
    crops_path = TRACKS / "crops" / "index.json"
    crops = json.loads(crops_path.read_text(encoding="utf-8")) if crops_path.exists() else {}
    robustness = TRACKS / "robustness.csv"
    return tracks, manifest, crops, pd.read_csv(robustness) if robustness.exists() else None


TRACKS_DATA, MANIFEST, CROPS, ROBUSTNESS = load()
ALL_TRACKS = TRACKS_DATA["tracks"]
BY_ID = {track["track_id"]: track for track in ALL_TRACKS}


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


def track_chunks(track: dict) -> list[dict]:
    start, end = track["t_start_ms"] / 1000, track["t_end_ms"] / 1000
    return [chunk for chunk in MANIFEST["chunks"] if chunk["t_start_s"] < end and start < chunk["t_end_s"]]


def show_crops(track: dict) -> None:
    items = sorted((CROPS[str(tracklet)] for tracklet in track["tracklets"] if str(tracklet) in CROPS), key=lambda item: item["t_s"])[:MAX_CROPS]
    if not items:
        return
    columns = st.columns(MAX_CROPS)
    for column, item in zip(columns, items):
        column.image(str(TRACKS / "crops" / item["file"]), caption=f"{camera_label(item['camera'])} {clock_text(item['t_s'])[3:8]}")


def show_links(track: dict) -> None:
    rows = link_rows(track)
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.caption("Трек видела одна камера, связей между камерами нет.")


def show_dossier(track: dict) -> None:
    st.subheader(f"Трек #{track['track_id']}")
    columns = st.columns(4)
    columns[0].metric("Начало", track["start"][:8])
    columns[1].metric("Конец", track["end"][:8])
    columns[2].metric("Длительность", f"{duration_s(track):.0f} с")
    columns[3].metric("Камеры", camera_numbers(track))
    st.write(f"Закрыт: {close_text(track)}.")
    for text in hint_texts(track):
        st.write(text)
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
        st.dataframe(pd.DataFrame(segment_rows(track)), hide_index=True, width="stretch")
        st.plotly_chart(plan_figure(track, PLAN_IMAGE, ROOT / "configs"), config=CHART_CONFIG, width="stretch", key=f"plan_{track['track_id']}")


def chosen_track(table: pd.DataFrame) -> int | None:
    """Трек выбирают строкой таблицы или точкой на таймлайне: действует то, что менялось последним."""
    rows = st.session_state.get("track_table", {}).get("selection", {}).get("rows", [])
    points = st.session_state.get("track_timeline", {}).get("selection", {}).get("points", [])
    from_table = int(table["Трек"].iloc[rows[0]]) if rows else None
    from_chart = int(points[0]["customdata"]) if points else None
    if from_chart != st.session_state.get("chart_choice"):
        st.session_state["chart_choice"] = from_chart
        if from_chart is not None:
            st.session_state["track"] = from_chart
    if from_table != st.session_state.get("table_choice"):
        st.session_state["table_choice"] = from_table
        if from_table is not None:
            st.session_state["track"] = from_table
    return st.session_state.get("track")


@st.fragment
def screen_tracks() -> None:
    table = pd.DataFrame(track_rows(ALL_TRACKS))
    selected = chosen_track(table)
    left, right = st.columns([5, 6])
    left.dataframe(table, hide_index=True, width="stretch", height=TIMELINE_HEIGHT, column_config=TABLE_COLUMNS,
                   on_select="rerun", selection_mode="single-row", key="track_table")
    right.plotly_chart(timeline_figure(ALL_TRACKS, selected), config=CHART_CONFIG, width="stretch", on_select="rerun", selection_mode="points", key="track_timeline")
    if selected is None:
        st.info("Выберите строку в таблице или нажмите на линию таймлайна, чтобы открыть досье трека.")
    else:
        show_dossier(BY_ID[selected])


def screen_returns() -> None:
    st.write("Отсутствие дольше минуты: слева ушедший трек, справа новый. Подсказка не меняет ID и подсчёт, вернувшийся получает новый номер. Кто вернулся быстрее минуты и уверенно узнан, сохраняет свой номер.")
    sheet = TRACKS / "gallery" / "hints.jpg"
    if sheet.exists():
        st.image(str(sheet))
    else:
        st.info("Лист ещё не собран: scripts/draw_gallery.py")


def screen_honesty() -> None:
    st.write(HONESTY_NOTE)
    for column, (label, value) in zip(st.columns(3), quality_tiles(TRACKS_DATA)):
        column.metric(label, value)
    if ROBUSTNESS is None:
        st.info("Проверка ещё не запускалась: scripts/check_robustness.py")
        return
    for column, (label, value) in zip(st.columns(3), robustness_tiles(ROBUSTNESS)):
        column.metric(label, value)
    for column, figure in zip(st.columns(2), robustness_curves(ROBUSTNESS)):
        column.plotly_chart(figure, config=CHART_CONFIG, width="stretch")

    st.subheader("Что будет, если отключить правило")
    st.dataframe(pd.DataFrame(rule_rows(ROBUSTNESS)), hide_index=True, width="stretch")

    st.subheader("Известные ошибки")
    for error in KNOWN_ERRORS:
        st.markdown(f"**{error.title}.** {error.text}")


SCREENS = {
    "Витрина": screen_showcase,
    "Вся запись": screen_recording,
    "Треки": screen_tracks,
    "Возвраты": screen_returns,
    "Как проверяли": screen_honesty,
}

if not PLAN_IMAGE.exists():
    shutil.copy(ROOT / "data" / "office.jpg", PLAN_IMAGE)
st.title("Сквозной трекинг людей по камерам")
st.caption(f"Сквозных треков: {TRACKS_DATA['total_tracks']}. Один номер, пока человека видит хотя бы одна камера и пока его нет меньше минуты. Нет дольше минуты: трек закрыт, при возврате новый номер.")
SCREENS[st.radio("Раздел", list(SCREENS), horizontal=True, label_visibility="collapsed")]()
