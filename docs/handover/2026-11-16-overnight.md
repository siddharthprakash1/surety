# Overnight 2026-11-16: report

> **Restored 2026-11-24.** Around 08:14 IST on 2026-11-16, after this report was written, the repository folder on the laptop was emptied (cause unknown; nothing was pushed, so the 41 local commits were lost). The branch was rebuilt on 2026-11-24 from the session record: source, tests and scripts from the session's own edits, docs by replaying every recorded edit onto `006f180` (the rebuilt README is byte-identical to the one written that night). The commit hashes in the table below refer to the lost history; the rebuilt branch carries the same content in fewer commits. Checks after the rebuild: `uv run pytest -q` gives 293 passed, 2 skipped, as at 07:05; every results table was regenerated from `artifacts/m1/` and matches the numbers below. See `docs/handover/2026-11-24-restore-and-research.md`.

Session: Claude Code on the Windows laptop (i7-11800H, 16 GB RAM, RTX 3050 Ti 4 GB, C: nearly full), ~00:55–08:00 IST.
All work is on the local branch `overnight-2026-11-16` (not pushed). `uv run pytest -q`: see the last section.

## TL;DR

- The project is set up on this laptop and reproduces M1 exactly (plan byte-for-byte, every table, fine-tune data stats).
- HANDOVER §5 step 1 is done: an independent re-review of the final fix wave found one incomplete fix (credentials stop) and nine more defects; all are fixed and tested. Tonight's own work then went through three more independent reviews, each scoped to the previous round's changes (the second found 10 defects, the third 9 plus minor ones, the fourth 1 high, 1 medium and several low; all fixed and tested; the final state is at the end of this file).
- New results, all CPU-only from the committed rows: uncertainty for H2 (risk control never exceeds ε on held-out sites over 200 calibration draws; the fitted threshold does in 200/200 at ε=0.05), an ε sweep figure, H3 re-read as inconclusive rather than refuted, a per-task-type analysis showing the held-out margin comes from pooling, and the first measurement of the critical-action detector (WebGuard: 16 % recall on high-risk actions from labels).
- M2 groundwork: the live harness ran in a real browser for the first time (MiniWoB++, scripted models) and found two real problems, both fixed: password fields were invisible to the credentials stop, and 42 % of MiniWoB++ tasks exposed 0–1 options because clickable spans/divs were ignored. Live gates are now fitted offline, calibrated live and updated online with audits; an H4 runner plays the six systems over 68 tasks in one command; the supervisor can be a hosted model with JSON-constrained answers; a Modal launcher for the Kev-4B fine-tune is ready.
- Not done: no model ran tonight. Serving Kev locally nearly filled the system disk twice (pagefile growth), so it was stopped and its 9 GB install removed. The fine-tune and the H3 re-run need your Modal account and a hosted model key.
- A tech-report draft is in `docs/report/draft.md`.

## What you need to decide or do

1. **Push?** `git push -u origin overnight-2026-11-16` (then merge to main if the review below holds). Nothing was pushed.
2. **Kev-4B fine-tune (Modal, ~$1–3 on an H100):** `uvx modal setup`, then `uvx --from modal==1.5.5 modal run scripts/modal_train_kev.py --max-steps 5` (smoke), then `--name webjev-v1`. Do not use Kev's own kev-finetune skill for this data: it keeps a 384-token state limit and silently drops ~94 % of WebJev records.
3. **H3 with a stronger supervisor:** any OpenAI-compatible endpoint, e.g. put the key in `OPENROUTER_API_KEY` (bash `export`, PowerShell `$env:OPENROUTER_API_KEY = '...'`), then `uv run python scripts/offline_supervisor.py --collect artifacts/m1/kev_compact.jsonl --eps 0.1 --n 100 --json --api-key-env OPENROUTER_API_KEY --base-url https://openrouter.ai/api --model <model> --max-tokens 2048 --out runs/h3_strong.jsonl`. ~190 calls of ~1.5k prompt tokens.
4. **Critical detector:** recall is 0.16 on WebGuard HIGH-risk actions from labels (0.24 with the URL). Adding keyword families (submit, sign up/register, save, download/install, Spanish verbs) would add +0.42 recall at +0.05 SAFE false flags (`results/webguard/detector.md`), but "submit" also escalates every MiniWoB submit. Not adopted; your call, or build the spec's Kev `noul` detector.
5. **This laptop:** do not serve Kev or Ollama models here (details below). Reboot once to give back ~3 GB of pagefile.

## Done, by commit

| commit | what |
|---|---|
| eb996cb | re-review fixes: credentials check independent of category order; payment/credential stems (submit order, CVV2, PIN, security/verification code, card no., expir*); select values classified; candidates in page order; one-element LAC escalations show the full list; executor stops traced; α=0.15 quantile pin; supervisor resume drops a torn last line |
| 055cfd1 | uncertainty: 200 calibration draws, site-cluster bootstrap, 200 adaptive streams (`results/m1/kev_compact_ci.md`) |
| 4872ab4 | README numbers that predated the fix wave refreshed (LAC 0.030, arisk ε=0.1 r=0.05 row, H3 answered rows 55/34) |
| 242d476 | WebGuard critical-detector recall (`results/webguard/detector.md`) |
| 031815c | supervisor `--json` (schema-constrained id) and `--api-key-env` for hosted endpoints |
| 5635798 | live gate: `loop.fit` fits T + gate on a collection's cal rows; runner applies T, updates the gate every step (escalation w=1, audit w=1/r, else w=0); H4 baselines; `run_miniwob.py --fit --audit-rate` |
| 7423975 | BrowserGym password inputs and unnamed fields from the DOM snapshot; first real-browser tests (`tests/integration/test_miniwob_live.py`) |
| de7d74e | `scripts/modal_train_kev.py` (Kev-4B fine-tune on Modal, `--max_state 7552`, init pinned to the revision M1 used) |
| 82b42e3 | H3 like-for-like line: both conditions answered, exact McNemar |
| b80100d | README / results doc / ledger rulings |
| 8671296 | ε sweep 0.02–0.20 (`results/m1/kev_compact_sweep.md`, `docs/results/kev_compact_sweep.png`) |
| 3cbc063 | tech-report draft (`docs/report/draft.md`) |
| aef684d | this report; HANDOVER pointer |
| ef2dd54 | H4 runner (`scripts/run_h4.py`, `src/surety/eval/live.py`): six systems × tasks × seeds, resumable, H4 table |
| cb8e1cf | clickable spans/divs/icons become options: MiniWoB++ tasks with no option 27 → 4 (`results/miniwob_observe.md`) |
| 029d4a3 | H4 task list (68 text-centric MiniWoB++ tasks) and safety list; README commands |
| ec5da38 | H4 runner writes a step trace per episode |
| 7725606 | second independent review of the branch, fixes: NONE answers are labels (acted on, learnt from); audit draws from one RNG per stream (seed 0 never audited before); live scoring uses the fit's 5/3 orderings; critical escalations show the full list; credential stems need context (PIN not "PIN code", expiry fields, MM/YY, cardnumber); autocomplete card/OTP/password inputs are secret; per_choice refused live; reasoning-model supervisor options; Modal launcher ships `modal_app` |
| 8344fb4 | marginal vs per-task-type risk control (`results/m1/kev_compact_groups.md`); docs aligned with the review |
| b84704e | shadow labels: evaluation-only supervisor labels on acted steps, so H4 reports live autonomous error per system |
| 156475d | Playwright env catches up with the BrowserGym findings (ARIA widgets, onclick, secret autocomplete, scripted dropdowns, navigation retry) |
| 1938365 | adaptive risk with one gate per task type on held-out streams |
| 2399c80 | third review: credential stems (card/ATM PIN codes, bare expiry, MM/YY not in DD/MM/YYYY, snake_case names); a NONE inside recovered reasoning is not an abstention |
| 915aa98 | third review: H4 resume restores α and the random streams (a resumed run equals an uninterrupted one), refuses other settings, retries errored episodes, stops after 3 failures in a row; matched comparison with supervisor-only; unanswered-call share; shadows on their own RNG; shared supervisor flags with 1,024 max tokens (64 starved gpt-oss) |
| 62491a7 | third review: labels fold into their controls (duplicate options removed), one option per bid, role nodes keep their names; H4 default 20 steps |
| 39d9322 | fourth review: a page-rejected action is an ordinary failed episode (it had been retried forever and dropped from the table); resume also checks backend, fit-file hash and supervisor; safer env/trace closing; inclusive 2-point margin |
| 4eaaa74 | fourth review: date/time inputs typed in ISO form again (enter-date, enter-time 3/3 with a scripted oracle); labels resolved before emission and folded only into form controls |
| 5677171 | fourth review: credential stems read camelCase/snake_case names; card-field synonyms; fewer date false positives |

Every ruling taken on your behalf is a line in `docs/handover/ledger/progress.md` (section "Overnight 2026-11-16"), with its cost if wrong.

## New results

- **H2 uncertainty.** Over 200 re-drawn calibration sets, risk@0.05 / risk@0.1 on held-out sites average 0.037 / 0.054 and exceed ε in 0 / 200 draws; fitted@ε exceeds it in 200 / 200 and 175 / 200; per-choice in 70 / 200 at ε=0.05. Site-bootstrap intervals (risk@0.05: 0.033 [0.018, 0.048]) are wider than the row-level Wilson intervals.
- **ε sweep.** For ε = 0.04–0.20 risk stays under ε in ≥ 99 % of draws, with automation within a point of per-choice; at ε = 0.02 it sits on target (0.021).
- **Adaptive risk** at audit rate 0.2 averages 0.048 / 0.094 over 200 streams but exceeds ε in 40 % / 35 % of them: it controls the long-run average, not each stream.
- **H3 inconclusive.** On the 26 escalated steps both conditions answered: full list 0.654 vs set 0.500, McNemar p = 0.22. The headline gap (0.326 vs 0.189) is mostly the set condition's higher parse-failure rate (58 % vs 37 %).
- **Safety.** The keyword override flags 16 % of WebGuard's HIGH-risk actions from the element label it sees live (24 % with the URL) and 2.6 % of SAFE ones.
- **Per task type.** Risk control's guarantee is marginal: on held-out sites element grounding sits at 0.059 at ε=0.05 (over ε in 70 % of draws), action prediction at 0.015. One gate per type holds on calibration sites but not on held-out sites (action prediction 0.074 / 0.118, over ε in 100 % / 89 % of draws): the marginal margin came from pooling. Adaptive risk with one gate per type (20 % audits) brings each type back to about ε on held-out streams (`results/m1/kev_compact_groups.md`).
- **Live harness** (scripted models, Chromium): login-user types the username and stops at the password field; click-test is solved without the supervisor.
- **MiniWoB++ coverage.** All 125 tasks open without error. Before tonight's env fix 27 tasks exposed no option and 26 only one (span links, clickable rows and icons have no ARIA role); now 4 and 19, the rest being drag/draw/coordinate tasks. Scripted oracle supervisors (no models) solve click-link (span links), click-button, click-widget, enter-text (type + submit), click-option, click-checkboxes and click-checkboxes-large (7–13 clicks) on 3/3 seeds each, so the new options are actionable and typing works in the browser; enter-text-2 fails only because the oracle does not change case.
- **H4 is one command** once Kev and a supervisor are up: `scripts/run_h4.py --tasks @configs/miniwob_h4.txt` (README, "H4 (MiniWoB++)"); add `--shadow-rate 0.2` to estimate each system's live autonomous error against the supervisor (evaluation calls, reported separately from the system's cost).

## Machine notes (this laptop)

- 15.7 GB RAM, 4 GB VRAM. C: had 19.5 GB free at the start and ~9 GB after tonight's installs (project env, data, Chromium, MiniWoB clone) and pagefile growth (12 → 17.4 GB, returned at reboot). At 02:25 C: showed 79 GB free: a separate Claude session ("Claude storage space on laptop", started ~02:16) was working on disk space then; this session deleted nothing beyond its own downloads (the Kev environment and model cache).
- Kev-0.8B was tried as a stand-in second executor (Clef is Apple-only). Its first request grew the Windows pagefile by ~4 GB within seconds (the GPU is too small and the driver spills to system memory); C: fell to 0.4–0.7 GB free twice before a watchdog stopped it. The Kev environment (7.4 GB) and model cache (1.7 GB) created tonight were deleted.
- An unrelated `vmmem` process holds ~4 GB of committed memory; `wsl --shutdown` did not stop it, so it was left alone.
- Quirks: no `wmic` (use PowerShell CIM); Hugging Face needs `HF_HUB_DISABLE_SYMLINKS=1` and `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` here (the stored HF token is invalid); `gh` is installed and logged in.

## State of the tree (07:05 IST)

- Branch `overnight-2026-11-16`: 41 commits on top of `006f180` (the handover), 55 files changed; working tree clean; nothing pushed.
- `uv run pytest -q`: 293 passed, 2 skipped (the browser tests); `SURETY_LIVE=1 MINIWOB_URL=... uv run pytest tests/integration`: 3 passed.
- Every M1 table regenerates from `artifacts/m1/` on CPU (`offline_analyze.py`, `offline_ci.py`, `offline_sweep.py`, `offline_groups.py`, `eval_webguard.py`).
- Reviews: four independent rounds tonight. The fourth round's fixes (39d9322, 4eaaa74, 5677171) are covered by tests and browser re-checks but were not themselves re-reviewed; a short scoped review of `f28bfb1..HEAD` is the first thing to do if the branch is going to main.
- A usage-limit pause stopped the session from about 03:30 to 06:55.
