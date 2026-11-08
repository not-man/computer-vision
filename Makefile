# Запускать из корня репозитория. Записи и план лежат в data/
PY := .venv/bin/python
PERCEPTION := artifacts/perception
TRACKS := artifacts/tracks
RENDER := render/full

.PHONY: setup demo check dense all tracks render robustness test

setup:
	uv sync --extra cpu

# Готовый результат: GPU не нужен, ничего не пересчитывается
demo: check
	.venv/bin/streamlit run app/demo.py

check:
	$(PY) scripts/demo_check.py

# Снимки внешности чаще раза в секунду: после детекции, один раз, на CPU около 10 минут
dense:
	$(PY) scripts/reembed.py data --every 3 --kinds box --out $(PERCEPTION) --suffix dense

# Всё после детекции: треки, мозаика, снимки, проверка устойчивости, витрина, отчёт
all: tracks render robustness
	$(PY) scripts/make_showcase.py
	$(PY) scripts/make_report.py
	$(PY) scripts/demo_check.py

tracks:
	$(PY) scripts/link_tracks.py $(PERCEPTION) --out $(TRACKS)
	mkdir -p results
	cp $(TRACKS)/tracks.csv results/tracks.csv
	cp $(TRACKS)/tracks_without_returns.csv results/tracks_without_returns.csv

render:
	$(PY) scripts/render_mosaic.py $(PERCEPTION) $(TRACKS) --out $(RENDER)
	$(PY) scripts/draw_gallery.py $(PERCEPTION) $(TRACKS)

robustness:
	$(PY) scripts/check_robustness.py $(PERCEPTION)

test:
	uv run --extra cpu pytest -q
