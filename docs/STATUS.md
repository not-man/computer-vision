# Состояние проекта

Читать первым после перерыва. Подробности и измерения в [DESIGN.md](DESIGN.md), раздел 0.

## Где что лежит
- Код в ветке `mct-pipeline`, на сервер не отправлялась. Автор в git не задан: коммитить с `-c user.name=... -c user.email=...` как у первого коммита.
- Записи и план: `../data`. Результат полного прохода в Colab: `artifacts/perception` и архив `../perception-20261106T213720Z-1-001.zip`.
- `artifacts/` и `render/` не в git и восстанавливаются командами ниже.
- Окружение: `uv sync --extra cpu` на машине без GPU.

## Как пересчитать
```bash
.venv/bin/python scripts/link_tracks.py artifacts/perception --out artifacts/tracks
.venv/bin/python scripts/render_mosaic.py artifacts/perception artifacts/tracks --out render/full
.venv/bin/python scripts/draw_gallery.py artifacts/perception artifacts/tracks
.venv/bin/python scripts/check_robustness.py artifacts/perception
uv run --extra cpu pytest -q
```
Склейка занимает секунды, рендер меньше 5 минут, галерея около 2–3 минут. Детекцию и ReID запускать только в Colab: `colab/run_perception.ipynb`, архив кода собирает `scripts/make_colab_bundle.py`.

## Решения пользователя
- Тяжёлые проходы только в Colab.
- В ходе работы давать список ручных проверок: что, где, когда.
- Время ожидания при уходе со всех камер не менять до ответа организаторов. Правило короткого возврата (10 с) просчитано, но не применено: DESIGN.md, раздел 0.
- Сначала точность, затем интерфейс. Точность сделана насколько это возможно без эталона.
- Коммитить на контрольных точках в локальную ветку, не пушить.

## Применено без явного ответа пользователя
- Запрет внешности на разрывах 0.15 вместо 0.05 (`gap_veto` в `src/mct/trinity.py`): пользователь сказал продолжать, не выбрав вариант. Откат одной строкой.

## Открытые вопросы к организаторам
Что считать длительным отсутствием; будет ли проверка на другой записи; остальное в DESIGN.md, раздел 16.
