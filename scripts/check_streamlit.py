"""Проверка показа: видео вне фрагмента не должно перезапускаться при выборе строки внутри него."""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

CHUNKS_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "render/check")
CHUNKS = sorted(CHUNKS_DIR.glob("*.mp4"))

st.set_page_config(layout="wide", page_title="Проверка показа")
choice = st.select_slider("Часть", options=[c.name for c in CHUNKS]) if len(CHUNKS) > 1 else CHUNKS[0].name
st.video(str(CHUNKS_DIR / choice))


@st.fragment
def track_table() -> None:
    table = pd.DataFrame({"track_id": range(1, 21), "камеры": ["cam_1, cam_2"] * 20})
    selected = st.dataframe(table, on_select="rerun", selection_mode="single-row", key="tracks", height=240)
    rows = selected.selection.rows
    st.write(f"Выбран трек: {table.track_id[rows[0]] if rows else 'нет'}")


track_table()
