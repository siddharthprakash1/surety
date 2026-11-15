# Task 10 report: M1 docs

Branch task-10, commit 2a8e0d8 (`docs: M1 results`). Not merged or pushed.

## Written
- `README.md` (replaced stub): summary, design paragraph, Results (M1) tables (H1 ECE, H2 static, H2 adaptive arisk + static risk reference rows, H3, ablation), one bullet per hypothesis, caveats, reproduce commands, status/next, license, credits.
- `docs/results/m1.md`: 10-bullet Setup, `results/m1/kev_compact.md` verbatim (checked with `diff`), "Reading the tables".
- `docs/results/kev_compact.png`: copy of the plot so the verbatim `![risk-coverage](kev_compact.png)` link resolves (results/**/*.png is gitignored; docs/ is not).

## Number sources
- All table values: `results/m1/kev_compact.md` (H1, H2, adaptive, ablation, H3, gold_in_set).
- "1495 vs 1988 tokens, 15.0 vs 19.9 s": ledger line 125 (rounded forms of the H3 table values).
- "~10% of grounding instances" (>64 options): ledger ruling on make_plan `--max-options 64`.
- Expectation-only guarantee / automation-maximising q̂: ledger CRC rulings.
- Setup facts: brief; cross-checked against `runs/plan.json` (seed 0, max_options 64, 600/400/600), `data/kev_serve.log` ("mlx (bfloat16)"), `runs/kev_ft/stats.json` (8000/500, 27 heldout sites), mean compact state_chars 3555 over all 1,600 rows.
- γ=0.02, tail=200, α clip [0.001, 0.5], weight rule: `src/surety/eval/analyze.py` and `src/surety/gate/policies.py` defaults (not in the listed sources).
- H3 answered rows 48 / 35: derived as n·(1−abstain−unparsed); confirmed against `runs/h3_0.1.jsonl`.

## Deviations from the brief (corrections)
- H1: brief says recalibration "halves" ECE. True only for test_in (0.167→0.086); test_shift 0.114→0.077, cal 0.150→0.095. README states the numbers instead.
- H3: brief says "n≈50 answered per condition". Actual answered rows are 48 (full) and 35 (set).
- H3: brief says the set is "25% cheaper among answered rows". Token/latency columns are means over all rows, not answered rows; README says "averaged over all rows".
- H2: added that per_choice also stayed within ε on test_shift (0.048, 0.070) with automation close to risk — needed for an honest H2 bullet.
- Kev-4B accuracy written as 0.453–0.485 (first_run, from H1) rather than "≈0.46–0.48".
- Reproduce: m1 advanced (task-8 fast-forward) right after the worktree was created at 2dcf106; task-10 was fast-forwarded to 180337a so `scripts/offline_supervisor.py` exists. The full-context ablation command uses `--n-perm 3 --splits test_in --limit 60`, inferred from the data (all 60 full rows have n_perm 3 and are the first 60 plan test_in ids). The WebJev download line (`uvx --from huggingface_hub hf download ...`) is a standard command, not the one actually logged.
- Commit trailer: `Co-Authored-By: Claude Opus 5.5` (harness attribution rule) instead of the brief's `Claude Fable 5.1`.

## Contradictions / gaps in the sources
- H3 n is 97 (full) / 103 (set), not 100 per condition: 97 ids in both conditions, 6 set-only. `data/h3.log` shows 6 full-list calls failed with HTTP 500 from Ollama; the file holds 103 unique ids although the final run selected 100, so 3 rows come from an earlier selection kept by resume. The two conditions are not fully paired.
- Verbatim results include the absolute local path `/Users/siddharthprakash/...` on the `collection:` line; consider trimming before publishing.
- `runs/kev_ft/trial1/training_config.json` exists (dated Nov 11), so a training trial was at least launched; docs follow the brief ("training not yet run").
- Spec's supervisor default (Qwen3.6-35B-A3B) differs from the M1 supervisor (gpt-oss:20b, per ledger ruling); docs say gpt-oss:20b.

`uv run pytest -q`: 156 passed, 1 skipped.
