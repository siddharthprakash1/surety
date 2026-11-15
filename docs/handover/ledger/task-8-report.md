# Task 8 report
Status: code DONE, real run BLOCKED (stopped after 2 lines).
Commit: 77798c9 on task-8 (worktree .claude/worktrees/task-8). Tests: 106 passed (incl. 2 new in tests/test_h3.py).

Built: scripts/offline_supervisor.py, select_escalated + h3_table in eval/analyze.py, --h3 in scripts/offline_analyze.py. --plan accepted but unused (ids come from the collection).
Selection: SplitConformal(0.1) fitted on cal debiased+T (q=0.977); 150 escalated rows selected.

First rows (runs/h3_0.1.aborted.jsonl): both chosen=null.
- full: prompt 1894, completion 152, 48.6 s, chosen null
- set: prompt 1852, completion 512 (hit max_tokens), 66 s, chosen null
Cause: gpt-oss:20b puts reasoning in message.reasoning; with max_tokens 512 content is empty/truncated, so _parse gets "". Trivial prompt works (content "A", 60 tokens). Latency 50-65 s/call, not 10-25 s, so 300 calls ~ 5 h.
Also: prompt-token savings from set are tiny here (1894 vs 1852) because the state text dominates.

Stopped run; no Ollama errors otherwise. Options: raise --max-tokens (e.g. 2048), or make Supervisor fall back to parsing `reasoning`, or lower reasoning effort via request param. Not changed without sign-off.

## Update: fix + relaunch
Commit 2abc570 (fix(supervise): reasoning level, reasoning-text fallback); 108 tests pass.
- Supervisor(reasoning=) prefixes system msg "Reasoning: <level>"; empty content falls back to last 200 chars of reasoning (Usage.fallback_parsed, logged per line); CLI --reasoning, --max-tokens default 1024.
- One-row check (reasoning low): full chosen=DONE (gold), 769 prompt/77 compl tok, 20.7 s; set chosen=None, 742/72 tok, 22.8 s. The set-condition model answered literally "NONE" (the system prompt permits abstaining), not a parse bug; counts as incorrect.
- Relaunched --n 100 --reasoning low --max-tokens 1024 -> runs/h3_0.1.jsonl, log data/h3.log.
- h3_table now has abstain and acc_given_answer columns (commit above, task-8).

## Fix round 1 (commit 81ebb61)
- F1: fallback parse takes the LAST id in the reasoning tail; skipped when finish_reason == "length".
- F2: Usage.abstained (explicit NONE only); h3_table reports abstain / unparsed / fallback as separate columns. Rows written before the fix lack `abstained`: a null `chosen` there counts as abstain (commented in code).
- F3: tests assert columns by name (parsed header), include gold_in_set=False rows, legacy-null row, fallback and NONE cases.
- Minors: selection test fits the policy (no hand-set q), seeded determinism + per-stratum counts, stray Mock assert removed, dead --plan flag dropped, resume tolerates a half-written last line, select_escalated stratifies on (split, task_type).
- Test: `uv run pytest -q` on a clean checkout of 81ebb61 (see result in handback). The run was stopped and resumed once at 10 lines; it later completed (200 lines) on its own.
- Caveat: task-8 worktree currently has an unresolved in-progress merge of m1 (conflicts in scripts/offline_analyze.py, src/surety/supervise/client.py, tests/test_supervisor.py); not mine, left untouched.

## Merge m1 (180337a) and final analysis
- Resolved client.py by routing reasoning prefix + reasoning-text fallback through m1's `_complete`; `choose` sets abstained and parses last id when fallback used; `slot_value`, `MockSupervisor(slots=)` kept; offline_analyze has `--ablation` and `--h3`.
- Fixed `test_h3_table_two_conditions` (stale last assert, failed on 81ebb61 in a clean checkout). `uv run pytest -q`: 156 passed, 1 skipped.
- Output: results/m1/kev_compact.md. Rows written before 81ebb61 lack `abstained` (null counted as abstain); n differs by condition (97 vs 103) because of skipped/errored calls and the resume.
## Ablation: compact vs full (`kev_full_test_in.jsonl`, shared ids, debiased)

| mode | n | acc | ECE | Brier | latency_ms | state_chars | n_perm |
|---|---|---|---|---|---|---|---|
| compact | 60 | 0.450 | 0.177 | 0.734 | 5842 | 3646 | 4.233 |
| full | 60 | 0.417 | 0.189 | 0.744 | 23963 | 13141 | 3.000 |


## H3: supervisor on full list vs conformal set

| cond | n | acc | abstain | unparsed | fallback | acc_given_answer | acc_gold_in_set | prompt_tok | completion_tok | latency_ms |
|---|---|---|---|---|---|---|---|---|---|---|
| full | 97 | 0.268 | 0.041 | 0.464 | 0.124 | 0.542 | 0.291 | 1988.309 | 253.155 | 19893.480 |
| set | 103 | 0.223 | 0.097 | 0.563 | 0.068 | 0.657 | 0.274 | 1494.845 | 239.981 | 14965.849 |

gold_in_set (LAC coverage on escalated rows): 0.816; action_prediction n=53, element_grounding n=50


