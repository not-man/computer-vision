# Результаты проверок

Журналы текущего результата, 61 трек. Методика и границы утверждений — [`docs/TEST_RESULTS.md`](../../docs/TEST_RESULTS.md). Метрики готового результата находятся в `artifacts/tracks/tracks.json` → `quality`; устойчивость — `artifacts/tracks/robustness.csv`.

- `pytest.xml`, `demo_check.log`, `make_all.log`, `eval_fragment.log`: тесты, 10 проверок, полный пересчёт, толпа и контрольные пары.
- `transition_audit.csv` (все 307 связей), `closure_audit.csv`, `track_audit.csv`, `missed_return_audit.csv`: вердикты просмотра глазами, в каждой строке указан проверяющий.
- `checks.json`: итоги просмотра, независимых точек, таблицы людей и толпы для раздела «Как проверяли». Собирает `scripts/summarize_checks.py`; он останавливается, если треки изменились после просмотра, а `make check` ловит устаревший файл.
- `experiments.csv`: опыты, которые не вошли в результат, и почему. Интерфейс и отчёт показывают эту таблицу как есть.
- `*_fixed_eval.log`, `frames_x960_eval.log`: сравнение прогонов детектора на фиксированной разметке толпы. Независимые точки: `evaluation/HOLDOUT.md`, таблица людей: `evaluation/people_truth.csv`.
- `audit/`: листы со снимками для этих таблиц, `audit/source_res/`: все треки снимками из исходных кадров. На снимках люди: как и записи, их нельзя выкладывать открыто.
- `history_64/`: журналы прежней версии с 64 треками.

Команды из корня: `python run.py check`, `python run.py test`, `python scripts/eval_fragment.py`, `python scripts/draw_audit.py transitions`. Листы закрытий: `python scripts/draw_audit.py closures`, это около пяти минут чтения видео.
