# Task 5 report: WebJev adapter and compiler

**Status:** DONE_WITH_CONCERNS
**Branch:** `worktree-agent-a6766dd2454ae5543` · **Commit:** `f13c29d feat(eval): webjev adapter`
**Tests:** `uv run pytest -q`: 10 passed (9 in `tests/test_webjev.py` plus the existing `test_types`)

## Files
- `src/surety/eval/__init__.py` (empty)
- `src/surety/eval/webjev.py`
- `tests/test_webjev.py`: the 6-row fixture is written to `tmp_path` by the test (no `tests/fixtures/` file); `make_plan` and `site_split` tests use synthetic `Instance`s.

Tests: the 6 named in the brief, plus `test_load_instances_filters_task_types`, `test_compact_keeps_goal_subgoal_and_referenced_elements` and `test_make_plan_fills_from_other_type_when_short`.

## Real data findings
- **`question` JSON keys** (structured rows):
  - action_prediction: `goal`, `rules` (`rules` is a str).
  - element_grounding: `goal`, `operation` (CLICK / TYPE_TEXT / SELECT), `rules` (`rules` is a list).
- **Goal and subgoal source:** `question["goal"]` is one string with fixed line prefixes, so I parse it line by line:
  - **goal:** the line starting `The user's original task (global constraints; no stage may drop any of its conditions): `. Present in 100% of structured rows.
  - **subgoal:** the line starting `[The only goal of the current stage] `. Present in 100% of rows and always a single line.
  - **Also kept:**
    - `Today is …`, the date line.
    - `Completed stages (…): [...]`, in 13% of AP rows and 6% of EG rows.
    - `Temporary semantic guidance after a stall (…): `, in 13% of AP rows and 6% of EG rows.
    - `The text helper explicitly rejected these fields …: `, in about 1% of rows.
  - **Dropped:** `rules` and the fixed boilerplate lines.
  - For EG rows, `question["operation"]` is emitted as `Operation: X`.
- **`context` JSON:**
  - Top-level keys: `page{url,title,text}`, `elements`, `recent_actions` (at most 10).
  - Element keys: `index` (str, 1-based, always present), `label`, `value`, `operations`. Optional: `role`, `expanded`, `checked`, `selected`, `context`, `position`, `_index`, `input_constraints`, `field_context`, `state_classes`, `options`, `row_context`.
- **Option formats:**
  - AP: `NAME: desc`. 2,035 options are region scrolls (`E72: Scroll down within the region: …`), and their numbers do not match any element index.
  - EG structured: `N: {json}`. `select_target` uses `N:M: {json}` (227 rows), giving id `E11:2`.
  - EG text-format (6,839 rows, about 21% of EG): `clickable: …` / `input: …` / `select: …`. These have no index, so every row repeats ids.
- **Size:** 64,122 rows (31,050 AP, all structured; 33,072 EG, of which 6,839 are text). All ids are unique. 1,443 sites.

## Implementation decisions
- **`load_instances`** recursively finds `*.parquet` files under `root`, skipping hidden dirs such as `.cache`. It filters rows by `task_type` through pyarrow and maps `website` to `site`. The full dataset loads in 1.2 s using about 2.5 GB RSS.
- **Compact state**, in order:
  1. Date, Task, Completed stages, Current subgoal, Guidance, Rejected fields
  2. Operation (EG only)
  3. URL, Title
  4. Recent actions, one per line: `- kind: action "text" (page changed)`
  5. Page text cut to `max_page_chars`
  6. Elements as `[idx] role | label | value=… | expanded=… | ops=…`
  - The element list keeps every element an option references (EG `N` / `N:M` → `N`). The remaining `max_elements − referenced` slots go to other elements in document order.
  - Element labels and values are cut to 100 characters (`_LABEL_CHARS`). Option labels are never cut.
- **EG option label:** `role | label | value=… | flags`, where label is the `element` string without its `[N] ` prefix.
- **Instructions**, keyed by `question_type`:
  - `operation`: "Which operation should the agent take next?"
  - `click_target`: "Which element should the agent click?"
  - `type_target`: "Which element should the agent type into?"
  - `select_target`: "Which option should the agent select?"
  - Anything else: "Which element should the agent act on?"
- **`make_plan`:** each split draws round-robin from per-task-type pools, which are seeded shuffles of the sorted ids. When one type runs short, the other fills the gap. cal and test_in are drawn from the same consumed pools, so they cannot overlap.
- **`site_split`:** sorts sites, shuffles with `Random(seed)`, and holds out `round(n_sites × holdout_frac)` sites.

## Real-data smoke run
- Every instance compiles. `gold_id == options[gold].id` holds, and ids are unique within each instance.
- Compact length in characters (p50 / p90 / max):

| Rows | Compact | Full |
|---|---|---|
| AP | 2,849 / 4,478 / 10,249 | 10.3k / 20.2k (p50 / p90) |
| EG structured | 3,935 / 6,340 / 12,981 | 15.6k / 29.3k (p50 / p90) |
| EG text (passed through) | 4,664 / 10,581 / 15,499 | same as compact |

- Plan with seed 0 and sizes 1500/1000/1500: 1,010 calibration sites and 433 held-out sites.
  - Every split is exactly half AP, half EG.
  - cal and test_in do not overlap.
  - Text-format rows make up 7–12% of each split.

## Concerns
1. **Text-format EG ids:** these get `clickable`, `clickable_2`, …, `clickable_N` (up to 137) from the generic `head: rest` rule plus dedupe. That is correct but unlike the `E12` style. If positional `E1..En` ids are preferred, it is a small change in `compile_instance`.
2. **Text-format rows are not compact:** both modes pass them through unchanged, and p90 is about 10.6k characters, above the ~1k-token target in the spec.
3. **Extra goal lines:** the compact state keeps completed stages, stall guidance and rejected fields along with goal and subgoal. These look decision-relevant (the LLM labeler saw them), but they go beyond "goal + subgoal" in the brief.
4. **Specific instructions:** grounding instructions use a verb taken from `question_type` rather than one fixed "act on" sentence.
5. **Commit trailer:** the commit ends with `Co-Authored-By: Claude Fable 5.1` as the task instructions specified. The session's system attribution says Opus 5.5, so amend if needed.
6. **Memory for Task 6:** `load_instances` on the full dataset holds about 2.5 GB, which matters for `make_plan.py`.
7. **Report location:** the worktree isolation guard blocked writing outside the worktree. This report is at `<worktree>/.superpowers/sdd/2026-11-06-surety-m1/task-5-report.md` (untracked, not committed) instead of the shared-checkout path.
