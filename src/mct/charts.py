"""Графики для интерфейса и офлайн-отчёта: один источник, чтобы они показывали одно и то же."""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml
from PIL import Image

from mct.trinity import TrinityConfig
from mct.video import clock_text

SURFACE = "#1a1a19"
TEXT = "#f0efec"
MUTED = "#c3c2b7"
GRID = "#3a3a37"
SERIES = "#3987e5"
# Цвет следует за камерой и не меняется при фильтрах; палитра проверена валидатором на тёмном фоне
CAMERA_COLORS = {
    "camera_new_1": "#3987e5", "camera_new_2": "#d95926", "camera_new_3": "#199e70",
    "camera_new_4": "#c98500", "camera_new_5": "#d55181",
}
PLAN_VIEW = (280.0, 700.0, 1135.0, 230.0)  # пиксели размеченного плана: x слева, x справа, y снизу, y сверху
TIMELINE_HEIGHT = 440
CLICK_STEP_S = 4.0


def camera_label(name: str) -> str:
    return f"cam_{name[-1]}"


def base_layout(height: int) -> dict:
    return {
        "height": height, "paper_bgcolor": SURFACE, "plot_bgcolor": SURFACE, "font": {"color": MUTED, "size": 13},
        "margin": {"l": 56, "r": 16, "t": 34, "b": 44}, "hoverlabel": {"bgcolor": "#262624", "font": {"color": TEXT}},
    }


def timeline_figure(tracks: list[dict], selected: int | None = None, height: int = TIMELINE_HEIGHT) -> go.Figure:
    """Все треки на общей шкале: сплошная линия по камерам, тонкий пунктир к подсказке о возврате."""
    order = {track["track_id"]: index for index, track in enumerate(tracks)}
    by_id = {track["track_id"]: track for track in tracks}
    figure = go.Figure()
    clickable: list[tuple] = []
    for camera, color in CAMERA_COLORS.items():
        xs, ys = [], []
        click_xs, click_ys, click_hover, click_ids = [], [], [], []
        for track in tracks:
            for segment in track["segments"]:
                if segment["kind"] != "observed" or segment["camera_id"] != camera:
                    continue
                start, end = segment["t_start_ms"] / 1000, segment["t_end_ms"] / 1000
                text = f"#{track['track_id']} · {camera_label(camera)} · {clock_text(start)[:8]}–{clock_text(end)[:8]}"
                xs += [start, end, None]
                ys += [order[track["track_id"]]] * 2 + [None]
                moments = [*np.arange(start, end, CLICK_STEP_S).tolist(), end]
                click_xs += moments
                click_ys += [order[track["track_id"]]] * len(moments)
                click_hover += [text] * len(moments)
                click_ids += [track["track_id"]] * len(moments)
        figure.add_scatter(x=xs, y=ys, mode="lines", name=camera_label(camera), line={"color": color, "width": 2}, hoverinfo="skip")
        clickable += list(zip(click_xs, click_ys, click_hover, click_ids))

    # Линии в Plotly не нажимаются: вдоль треков идут почти невидимые точки, на них подсказка и выбор трека
    if clickable:
        xs, ys, hover, ids = zip(*clickable)
        figure.add_scatter(x=xs, y=ys, mode="markers", marker={"size": 10, "color": TEXT, "opacity": 0.01}, hovertext=hover, hoverinfo="text",
                           customdata=ids, showlegend=False)

    xs, ys, hover = [], [], []
    for track in tracks:
        for hint in track["possibly_same_person_as"]:
            old = by_id.get(hint["track_id"])
            if old is None:
                continue
            xs += [old["t_end_ms"] / 1000, track["t_start_ms"] / 1000, None]
            ys += [order[old["track_id"]], order[track["track_id"]], None]
            text = f"#{track['track_id']} похож на #{old['track_id']}, сходство {hint['r']:.2f}"
            hover += [text, text, None]
    if xs:
        figure.add_scatter(x=xs, y=ys, mode="lines", name="подсказка о возврате", line={"color": MUTED, "width": 1, "dash": "dot"},
                           hovertext=hover, hoverinfo="text")
    if selected in order:
        figure.add_hline(y=order[selected], line={"color": TEXT, "width": 1})

    starts = [track["t_start_ms"] / 1000 for track in tracks]
    ends = [track["t_end_ms"] / 1000 for track in tracks]
    first, last = (min(starts), max(ends)) if tracks else (0.0, 60.0)
    marks = list(range(int(first // 120 + 1) * 120, int(last), 120))
    figure.update_layout(
        **base_layout(height),
        legend={"orientation": "h", "y": 1.0, "yanchor": "bottom", "x": 0, "font": {"color": TEXT}},
        xaxis={"tickvals": marks, "ticktext": [clock_text(mark)[:5] for mark in marks], "gridcolor": GRID, "zeroline": False, "side": "bottom"},
        yaxis={"autorange": "reversed", "showticklabels": False, "showgrid": False, "zeroline": False, "title": {"text": "треки по времени начала"}},
    )
    return figure


def plan_figure(track: dict, plan_image: Path, configs: Path = Path("configs")) -> go.Figure:
    """Путь одного трека на плане: сплошная там, где человека видели, пунктир там, где нет."""
    features = yaml.safe_load((configs / "plan_features.yaml").read_text(encoding="utf-8"))
    scale, placement = features["plan_px_per_m"], features["plan_image"]
    path = pd.DataFrame(track["plan_path"], columns=["x", "y", "t_ms"])
    figure = go.Figure()
    image = Image.open(plan_image)
    figure.add_layout_image(
        source=image, xref="x", yref="y", x=placement["dx"], y=placement["dy"], sizing="stretch", layer="below", opacity=0.75,
        sizex=image.width * placement["scale"], sizey=image.height * placement["scale"], xanchor="left", yanchor="top",
    )
    if len(path):
        pieces = (path.t_ms.diff() > 1000).cumsum()
        previous = None
        for _, piece in path.groupby(pieces):
            if previous is not None:
                figure.add_scatter(x=[previous.x * scale, piece.x.iloc[0] * scale], y=[previous.y * scale, piece.y.iloc[0] * scale],
                                   mode="lines", line={"color": SERIES, "width": 2, "dash": "dot"}, hoverinfo="skip", showlegend=False)
            hover = [clock_text(value / 1000)[:8] for value in piece.t_ms]
            figure.add_scatter(x=piece.x * scale, y=piece.y * scale, mode="lines", line={"color": SERIES, "width": 3},
                               hovertext=hover, hoverinfo="text", showlegend=False)
            previous = piece.iloc[-1]
        for point, label in ((path.iloc[0], "начало"), (path.iloc[-1], "конец")):
            figure.add_scatter(x=[point.x * scale], y=[point.y * scale], mode="markers", hoverinfo="skip", showlegend=False,
                               marker={"size": 10, "color": SERIES, "line": {"color": SURFACE, "width": 2}})
            figure.add_annotation(x=point.x * scale, y=point.y * scale, text=label, showarrow=False, yshift=18,
                                  bgcolor=SURFACE, borderpad=3, font={"color": TEXT})
    left, right, bottom, top = PLAN_VIEW
    figure.update_layout(
        **{**base_layout(560), "margin": {"l": 0, "r": 0, "t": 0, "b": 0}},
        xaxis={"range": [left, right], "visible": False}, yaxis={"range": [bottom, top], "visible": False, "scaleanchor": "x"},
    )
    return figure


def curve_figure(values: pd.Series, tracks: pd.Series, title: str, x_title: str, current: float) -> go.Figure:
    """Число треков от одного порога; линия одна, поэтому легенды нет, а текущее значение подписано."""
    figure = go.Figure()
    figure.add_scatter(x=values, y=tracks, mode="lines+markers", line={"color": SERIES, "width": 2}, marker={"size": 8, "color": SERIES},
                       hovertemplate=f"{x_title}: %{{x}}<br>треков: %{{y}}<extra></extra>", showlegend=False)
    chosen = tracks[values == current]
    if len(chosen):
        figure.add_annotation(x=current, y=int(chosen.iloc[0]), text=f"сейчас: {int(chosen.iloc[0])}", showarrow=True, arrowcolor=MUTED,
                              ax=30, ay=-32, font={"color": TEXT})
    figure.update_layout(
        **base_layout(300), title={"text": title, "font": {"color": TEXT, "size": 15}, "x": 0},
        xaxis={"title": {"text": x_title}, "gridcolor": GRID, "zeroline": False},
        yaxis={"title": {"text": "треков"}, "gridcolor": GRID, "zeroline": False, "rangemode": "tozero"},
    )
    return figure


def robustness_curves(robustness: pd.DataFrame) -> list[go.Figure]:
    def curve(group: str, title: str, x_title: str, current: float) -> go.Figure:
        rows = robustness[robustness.group == group].reset_index(drop=True)
        return curve_figure(rows.name.str.split().str[1].astype(float), rows.tracks, title, x_title, current)

    config = TrinityConfig()
    return [curve("кривая длительности", "Число треков зависит от минимальной длительности", "минимальная длительность трека, с", config.min_track_s),
            curve("кривая r_veto", "Число треков зависит от запрета по внешности", "порог R_veto", config.r_veto)]
