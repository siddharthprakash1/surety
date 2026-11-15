# Task 2 report
Status: DONE. Branch task-2.
- Built `src/surety/decide/{__init__,base,mock,systemone,debias,recalibrate}.py`, `tests/test_decide.py` (11 tests).
- `uv run pytest -q` -> `11 passed`.
- Tests written first (collection failed on missing module), then implemented.
- Self-review: 404 on permute cached only when `permute_endpoint is None`; explicit True re-raises. Permute run order is not used (probs keyed by id). Extra tests beyond brief: noul, noul_debiased, auth header.

## Fix round 1
- `SystemOneClient`: added `close()`, `__enter__`, `__exit__`.
- Tests: temperature `2.0 < t < 4.5`; permute path asserts `input_tokens == 7`; new uniform-fallback test (`{"z":1}`); new context-manager test; `noul_debiased` import moved to top.
- `uv run pytest tests/test_decide.py -q` -> `12 passed`.

## Fix round 2
- `TemperatureScaler.fit`: NLL now computed per row over that row's own ids (log-sum-exp); no id union, no 1e-6 filler.
- Added `test_temperature_fit_is_invariant_to_per_row_ids` (also checks `apply` on unseen ids).
- `uv run pytest tests/test_decide.py -q` -> `13 passed`.
