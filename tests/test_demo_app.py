"""Экраны интерфейса на настоящем результате. Без готовых файлов тесты пропускаются: их создаёт make all."""
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
TRACKS = ROOT / "artifacts" / "tracks" / "tracks.json"
pytestmark = pytest.mark.skipif(not (TRACKS.exists() and (ROOT / "render" / "full" / "manifest.json").exists()), reason="нет готового результата")


def opened(screen: str) -> AppTest:
    app = AppTest.from_file(str(ROOT / "app" / "demo.py"), default_timeout=60).run()
    return app.radio[0].set_value(screen).run()


@pytest.mark.parametrize("screen", ["Витрина", "Вся запись", "Треки", "Возвраты", "Как проверяли"])
def test_every_screen_opens_without_errors(screen: str) -> None:
    assert not opened(screen).exception


def test_total_count_is_on_the_screen() -> None:
    total = json.loads(TRACKS.read_text(encoding="utf-8"))["total_tracks"]
    assert any(f"Сквозных треков: {total}." in caption.value for caption in opened("Треки").caption)


def test_known_errors_are_shown() -> None:
    app = opened("Как проверяли")
    assert any("Двое в обнимку" in block.value for block in app.markdown)
