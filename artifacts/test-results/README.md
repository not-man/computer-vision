# Результаты проверок

Здесь сохраняются фактические журналы нового прогона. Методика и границы утверждений — [`docs/TEST_RESULTS.md`](../../docs/TEST_RESULTS.md). Метрики готового результата находятся в `artifacts/tracks/tracks.json` → `quality`; устойчивость — `artifacts/tracks/robustness.csv`.

Команды из корня: `python run.py check`, `python run.py test`, `python scripts/eval_fragment.py`. До нового прогона отсутствие журнала не означает успешную проверку.
