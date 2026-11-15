# Task 6 report
- Branch task-6, commit 0cb15ca `feat(eval): collection scripts`. Files: src/surety/eval/collect.py, scripts/make_plan.py, scripts/offline_collect.py, tests/test_collect.py.
- Tests: 42 passed (4 new: write, resume, no-resume overwrite, skip backend error).
- Smoke (kev, compact, n_perm 5, 6 rows, plan_smoke 20/10/10):
  | type | n_opt | latency ms | input_tokens | state_chars | top==gold | top p |
  | AP | 8 | 4180 | 0 | 4437 | N | .37 |
  | EG | 64 | 46992 | 0 | 7330 | Y | .15 |
  | AP | 7 | 2681 | 0 | 3395 | Y | .21 |
  | EG | 20 | 3494 | 0 | 3273 | Y | .22 |
  | AP | 7 | 2423 | 0 | 3058 | Y | .41 |
  | EG | 32 | 59268 | 0 | 12954 | Y | .17 |
- probs keys == option_ids on all 6; accuracy 5/6; truncated none.
- Concerns: input_tokens is 0 (server's permute endpoint reports no usage); latency scales with n_options (47-59 s for 32-64 options at n_perm 5) -> full run is slow; `clef` backend is just SystemOneClient with name=clef (pass --base-url/--model); CLI exits 1 when a fully-resumed run writes 0 rows.

## Fix round 1
- collect.py: resume tolerant of torn lines (truncates to last newline, skips bad JSON); `done` keyed on (id, mode, backend) with stderr warning on mismatch; aborts (CollectAborted) after 20 consecutive failures; final "written N, skipped M"; progress counts skipped; `n_perm_large`/`large_options` -> row `n_perm`.
- offline_collect.py: round-robin across splits before `--limit`; validates split names and `--root` vs plan; `--n-perm-large` (3), `--large-options` (16); exit 1 only on abort; 0 rows from a full resume exits 0 (all-fail raises abort/exit... see below).
- make_plan.py: `--max-options`, recorded as `max_options`.
- Tests: `uv run pytest tests/test_collect.py -q` -> 8 passed; full suite 46 passed. Mock CLI check: --limit 6 gives cal,test_in,test_shift,cal,test_in,test_shift.
- R4 follow-up: `collect` raises CollectAborted when written == 0 and skipped > 0 (CLI exits non-zero); fully-resumed run (0 written, 0 skipped) still exits 0.
- Tests: `uv run pytest tests/test_collect.py -q` -> 9 passed; full suite 47 passed.
