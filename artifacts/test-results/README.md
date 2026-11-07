# Результаты проверок

Журналы текущего результата, 61 трек. Методика и границы утверждений — [`docs/TEST_RESULTS.md`](../../docs/TEST_RESULTS.md). Метрики готового результата находятся в `artifacts/tracks/tracks.json` → `quality`; устойчивость — `artifacts/tracks/robustness.csv`.

- `pytest.xml`, `demo_check.log`, `make_all.log`, `eval_fragment.log`: тесты, 10 проверок, полный пересчёт, толпа и контрольные пары.
- `transition_audit.csv`, `closure_audit.csv`, `track_audit.csv`, `missed_return_audit.csv`: вердикты просмотра глазами, в каждой строке указан проверяющий.
- `audit/`: листы со снимками для этих таблиц, `audit/source_res/`: все треки снимками из исходных кадров. На снимках люди: как и записи, их нельзя выкладывать открыто.
- `history_64/`: журналы прежней версии с 64 треками.

Команды из корня: `python run.py check`, `python run.py test`, `python scripts/eval_fragment.py`, `python scripts/draw_audit.py transitions`. Листы закрытий: `python scripts/draw_audit.py closures`, это около пяти минут чтения видео.
