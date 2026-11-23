# HANDOVER — Surety (read this first)

**Update 2026-11-24:** read `docs/handover/2026-11-24-restore-and-research.md` first: the branch was lost on 2026-11-16 and rebuilt; new gates on a learned error score; research and a ranked roadmap in `docs/research/2026-11-24-improvements.md`; an independent review's fixes.

**Update 2026-11-16:** an overnight session on the owner's Windows laptop continued from here on branch `overnight-2026-11-16`. Read `docs/handover/2026-11-16-overnight.md` first: §5 step 1 (re-review of the fix wave) is done and its findings fixed; new uncertainty, ε-sweep, H3 and WebGuard results; live-harness and Modal groundwork; what needs the owner (Modal account, hosted supervisor key, push). The rest of this file is the original M1 handover.

Written 2026-11-15 by the Claude Code session that built M1. You are the next agent/operator. Everything below is what you need to continue without the original conversation.

## 1. What this project is

Surety is an open-source web computer-use-agent harness with **guaranteed escalation**: a small local decision model (an open Jev-style "System One" model, Kev-4B) picks each step's action from an enumerated list of page elements; a reasoning LLM supervises; a gate decides who acts. The gate's job is to keep **wrong autonomous actions ≤ ε per step**, including on websites never seen during calibration, and to adapt online from escalations plus random audits.

Owner's goals (in priority order): a reproducible benchmark result + tech report (master's applications, resume), then an open-source release, then maybe a startup. Prior art and the gap are in the spec. Style rule from the owner: aggressively concise in code, docs, commits; comments only where code cannot say it.

Read in this order: `README.md` (results + reproduce), `docs/superpowers/specs/2026-11-06-surety-design.md` (design, hypotheses H1–H4), `docs/superpowers/plans/2026-11-06-surety-m1.md` (task plan), `docs/handover/ledger/progress.md` (every decision/ruling made during execution, deferred minors), `docs/results/m1.md` (all tables + definitions).

## 2. Current state (branch `main` == `m1` == `final-fix`, HEAD at the "chore: ignore .claude/" commit plus the handover commit)

- Code: `src/surety/{types,calib,decide,gate,supervise,eval,env,compile,loop,trace}` — all M1 tasks (0–8, 11, 12) implemented, reviewed task-by-task, merged. `uv run pytest -q` → **183 passed, 1 skipped** (the skipped test is the live Playwright test, needs `SURETY_LIVE=1`).
- The last seven commits (`f9a3dc3..85aa7e5`, "final fix wave") address the whole-branch review's findings (1 Critical + 5 Important + 7 Minor; list in §6). They were implemented and tested but the **scoped re-review of that fix wave did not happen** (the session hit a usage limit). Treat them as unreviewed: do that re-review first (see §5 step 1).
- Experiments done (artifacts committed under `artifacts/m1/`, regenerable commands in README):
  - `plan.json` — 600 cal / 400 test_in (calibration sites) / 600 test_shift (held-out sites), ≤64 options, seed 0, 30% of sites held out.
  - `kev_compact.jsonl` — Kev-4B zero-shot, compact state, 5 permutations (3 above 16 options), all 1,600 rows.
  - `kev_full_test_in.jsonl` — full-context ablation, 60 test_in rows, 3 permutations.
  - `h3_0.1.jsonl` (first H3 run, unpaired, partly with a rejected parser) and `h3_0.1b.jsonl` (**paired re-run, use this one**): gpt-oss:20b via Ollama, "Reasoning: low", max_tokens 1024, 100 escalated rows (LAC set at ε=0.1), paired n=95.
  - `kev_ft_stats.json` — stats of the Kev fine-tune data (8,000 train / 500 held-out on 27 whole sites; the JSONL files are regenerable in seconds with `scripts/make_kev_train.py`, not committed).
- Results tables: `results/m1/kev_compact.md` (regenerated with the fixed code and the paired H3) and the copy in `docs/results/m1.md`. README tables were refreshed from it at handover.
- Not done / not run: Kev fine-tune training (5-step trial on Apple MPS logged no step in 20 min — transformers reference kernels; use Kev's Modal H100 recipe, ~$1 per run), Clef-flash second executor (server launch was blocked by the auto-mode classifier in the original session; `scripts/serve_clef.sh 8010` is ready), any live browser run (harness skeleton exists, `scripts/run_miniwob.py`), Mind2Web, WebGuard detector recall, H4.

## 3. Headline numbers (M1, Kev-4B zero-shot on WebJev; details in README / docs/results/m1.md)

- Accuracy ≈ 0.48 in-dist / 0.46 held-out sites; mean confidence 0.23–0.40 (weak but roughly calibrated).
- H1: temperature refit halves ECE (test_in 0.167 → 0.086; test_shift 0.114 → 0.077); permutation debiasing alone does not help.
- H2: conformal risk control (`risk@ε`) meets ε on held-out sites (0.033 @ ε=.05, 0.047 @ ε=.1; automation 0.22 / 0.28); the empirically fitted threshold drifts over target there (0.060 / 0.105); LAC coverage sets are safe but automate only 3–5%. Adaptive risk control needs audits: r=0 drifts to α=0.5, r=0.2 stays within ε and automates a bit more than static (0.247 / 0.320).
- H3 (paired re-run, n=95): **did not hold** with this supervisor — set condition acc 0.189 vs full 0.326; even among answered rows 0.529 vs 0.564; prompt tokens −19% (1371 vs 1700), latency −16%. 37–58% of gpt-oss replies unparseable even at low reasoning. Needs a better supervisor (see §5).
- Ablation (60 rows): compact state 0.450 acc / 5.8 s vs full context 0.417 / 24 s (n_perm mismatch caveat).

## 4. How to set up on the new machine (Apple Silicon assumed; Linux/CUDA works for Kev too)

```bash
git clone <this repo> surety && cd surety
uv sync --group dev && uv run pytest -q                      # expect 183 passed, 1 skipped
uvx --from huggingface_hub hf download Lexmount/WebJev --repo-type dataset --local-dir data/webjev   # 120 MB
git clone https://github.com/jaredpalmer/kev third_party/kev  # gitignored
(cd third_party/kev && uv sync --extra serve && uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b --port 8009)  # downloads ~8 GB base
# supervisor: any OpenAI-compatible chat endpoint; M1 used Ollama gpt-oss:20b at http://127.0.0.1:11434 (--reasoning low)
# optional second executor: scripts/serve_clef.sh 8010   (Clef-flash, MLX 4-bit, 6 GB; same /v1/systemone API)
```
Regenerate every table without re-collecting: `uv run python scripts/offline_analyze.py --collect artifacts/m1/kev_compact.jsonl --ablation artifacts/m1/kev_full_test_in.jsonl --h3 artifacts/m1/h3_0.1b.jsonl --out results/m1/kev_compact.md`.
Disk: Kev base ~8 GB, Clef 6 GB, WebJev 0.12 GB, Kev venv ~4 GB. RAM: 48 GB Mac ran Kev (8–16 GB) + Ollama gpt-oss (12 GB) together; do not co-load more.

## 5. What to do next, in order

1. **Re-review the final fix wave** (`git diff 2a8e0d8..85aa7e5`), findings list in §6; fix anything wrong; then the branch is mergeable. Known leftover from that wave: README/docs "Reading the tables" should state that T and the gates are fitted on the same cal rows (in-sample for T) — check it is there.
2. **Fine-tune Kev on WebJev** (the fine-tune trigger fired: automation ≤ 0.34 at zero-shot). `scripts/make_kev_train.py` (data) + `scripts/train_kev.sh` (flags) exist; run training on a GPU box or Kev's Modal recipe (see `third_party/kev/README.md` "Fine-tune", `--init_from jaredpalmer/kev-4b --max_state 7552`); then serve the checkpoint and re-run `scripts/offline_collect.py` on the same `artifacts/m1/plan.json` (identical rows → directly comparable tables). Calibration rows (`cal`) and all test rows are disjoint from the training data by construction — keep it that way.
3. **H3 with a stronger supervisor** (a 30B+ reasoning model or a hosted one): `scripts/offline_supervisor.py --reasoning medium`, check the `unparsed` column stays < 10%; if the set condition still loses, the H3 hypothesis is falsified for the write-up (that is a fine result).
4. **Clef-flash as a second executor** (`scripts/serve_clef.sh`, then `offline_collect.py --backend clef --base-url http://127.0.0.1:8010`), same plan.
5. **Live runs (M2)**: `scripts/run_miniwob.py` (MiniWoB++ via BrowserGym; setup lines printed by the script), then WebArena-Verified hard subset (BrowserGym has `webarena_verified_hard` on main since 2026-10-27; images are amd64-only → emulation on a Mac or a Linux box). Fit the gate on offline cal rows, run six systems (supervisor-only, executor-only, fixed threshold, Jev-Mobile rule, risk, adaptive risk) → H4.
6. Mind2Web (human labels; test splits are password-protected, train split usable) and WebGuard (critical-detector recall) — spec §Safety.
7. Multiple seeds + confidence intervals; then the tech report / blog post. Original milestone dates have slipped: set new ones.

## 6. The final-review findings the last seven commits addressed (verify these)

Critical: runner's payment/critical check only ran on supervisor picks → now every planned action is classified; unwrapped policies are wrapped in `CriticalOverride`. Important: `alpha_raw` unbounded above (now upper-clipped at `alpha_max`, lower side unclipped); reasoning-text fallback leaked into `slot_value` (now returns None on fallback, one-line cap); `RiskConformal.decide` escalated with the CRC set (now escalates with the LAC set at the same ε; `prediction_set` stays CRC for analysis); `h3_table` unpaired (now paired on ids in both conditions); no credentials stop (now role `password`, `credentials` detector category, `stop:credentials`). Minor: k-th order-statistic quantile (was (k+1)-th, conservative); Wilson 95% CI column for auto_err; userinfo stripped from traced URLs; `.claude/` ignored; `gold_in_set` label; H3 set listed in page order. Deferred minors per task are in `docs/handover/ledger/progress.md` (lines starting `Task N: minor (deferred)`).

## 7. Every ruling made on the owner's behalf (so you can undo any)

See the `Ruling:` lines in `docs/handover/ledger/progress.md`. The consequential ones: gate target switched from coverage (LAC) to **conformal risk control on the autonomous-error loss** (fitted on the monotone envelope, automation-maximising threshold); adaptive update applies the ε drift every step with weight 0 on unaudited singletons; instances with >64 options excluded from M1; 3 permutations above 16 options; supervisor = gpt-oss:20b via Ollama instead of the spec's Qwen3.6-35B (disk); Kev fine-tune on Apple MPS abandoned for cloud; WebGuard, Mind2Web, Clef deferred; critical-action keywords match stems; conformal quantile ∞ (full set) when infeasible; PerChoiceThreshold global fallback below 30 items; T and gates fitted on the same cal rows (documented).

## 8. Process used (so you can keep the same quality bar)

Superpowers skills: brainstorming → spec → plan → subagent-driven development (one implementer per task in its own git worktree, a task reviewer, scoped re-reviews per fix round, a final whole-branch review). Commit messages end with a `Co-Authored-By` trailer. Tests are one-line-named, numbers pinned. Keep the ledger habit: one line per decision with "cost if wrong".

## 9. Things that bit us (avoid)

- zsh does not word-split unquoted `$VAR` — write paths literally in `git diff -- a b c`.
- Kev refuses to overwrite an existing run dir — never `mkdir` the `--out` dir first.
- gpt-oss spends `max_tokens` on hidden reasoning → empty `content`; prefix the system prompt with `Reasoning: low` and keep `max_tokens` ≥ 1024, or use a non-reasoning supervisor.
- Kev latency scales with option count (32–64 options → 30–60 s per decision on an M4 Pro); keep states compact (~1k tokens) and cap options.
- Ollama keeps models resident (keep_alive) — budget GPU memory accordingly.
