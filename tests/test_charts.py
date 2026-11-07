from pathlib import Path

import pandas as pd
from PIL import Image

from mct.charts import curve_figure, plan_figure, timeline_figure


def track(track_id: int, start_s: float, end_s: float, hints: list[dict] | None = None) -> dict:
    return {
        "track_id": track_id, "t_start_ms": int(start_s * 1000), "t_end_ms": int(end_s * 1000),
        "segments": [{"kind": "observed", "camera_id": "camera_new_1", "t_start_ms": int(start_s * 1000), "t_end_ms": int(end_s * 1000)}],
        "possibly_same_person_as": hints or [],
        "plan_path": [[10.0, 10.0, 0], [11.0, 10.0, 500], [15.0, 10.0, 5000], [16.0, 10.0, 5500]],
    }


def test_timeline_keeps_one_trace_per_camera_and_marks_hints() -> None:
    tracks = [track(1, 0.0, 10.0), track(2, 100.0, 120.0, [{"track_id": 1, "r": 0.8, "absent_s": 90.0}])]
    figure = timeline_figure(tracks)
    names = [trace.name for trace in figure.data]
    assert names[:5] == ["cam_1", "cam_2", "cam_3", "cam_4", "cam_5"]
    assert "подсказка о возврате" in names
    hint = next(trace for trace in figure.data if trace.name == "подсказка о возврате")
    assert list(hint.x[:2]) == [10.0, 100.0]


def test_camera_color_does_not_depend_on_which_tracks_are_shown() -> None:
    colors = [trace.line.color for trace in timeline_figure([track(1, 0.0, 10.0)]).data[:5]]
    assert colors == [trace.line.color for trace in timeline_figure([]).data[:5]]


def test_selected_track_is_marked() -> None:
    assert len(timeline_figure([track(1, 0.0, 10.0)], selected=1).layout.shapes) == 1
    assert len(timeline_figure([track(1, 0.0, 10.0)]).layout.shapes) == 0


def test_plan_path_is_dotted_across_the_pause(tmp_path: Path) -> None:
    image = tmp_path / "plan.jpg"
    Image.new("RGB", (612, 658), "white").save(image)
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "plan_features.yaml").write_text("plan_px_per_m: 29.0\nplan_image: {scale: 2.0, dx: 4.0, dy: 2.0}\n", encoding="utf-8")
    figure = plan_figure(track(1, 0.0, 10.0), image, configs)
    dotted = [trace for trace in figure.data if trace.line.dash == "dot"]
    assert len(dotted) == 1
    assert list(dotted[0].x) == [11.0 * 29.0, 15.0 * 29.0]


def test_curve_marks_the_current_value() -> None:
    figure = curve_figure(pd.Series([1.0, 2.0, 3.0]), pd.Series([116, 108, 102]), "заголовок", "порог", 2.0)
    assert "108" in figure.layout.annotations[0].text
    assert figure.data[0].showlegend is False
