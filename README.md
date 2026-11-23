# Surety

Surety is a web computer-use agent harness. A small local decision model picks each step's action from an enumerated list, and a reasoning LLM supervises. A conformal gate decides who acts, keeping wrong autonomous actions at or below a target ε per step, and adapts online from escalations and random audits as sites change.

## Design

Per step, the executor (Kev-4B, served locally) scores the options in several shuffled orders (debias), the averaged probabilities are temperature-scaled on calibration data (recalibrate), and a gate either acts on one option or escalates to the supervisor (any OpenAI-compatible chat model) with a candidate set. Gate families on identical code: fixed, margin, fitted and per-choice probability thresholds; split-conformal LAC sets (coverage control); conformal risk control on the autonomous-error loss 1{|C|=1 ∧ gold∉C}; adaptive versions of the last two that update α online from escalated steps plus a random audit of acted steps (weight 1/r). A critical-action override (pay, send, delete, account changes) wraps every policy: payment stops the episode, the other categories escalate with the full option list; typing into or selecting in password, card or other credential fields stops the episode. Spec: `docs/superpowers/specs/2026-11-06-surety-design.md`.

## Results (M1)

Offline, on [WebJev](https://huggingface.co/datasets/Lexmount/WebJev): Kev-4B zero-shot, compact state, 600 cal / 400 test_in (calibration sites) / 600 test_shift (held-out sites). Full tables, setup and definitions: [`docs/results/m1.md`](docs/results/m1.md).

**H1: ECE (15 bins)**

| split | first_run | debiased | debiased+T |
|---|---|---|---|
| cal | 0.150 | 0.160 | 0.095 |
| test_in | 0.167 | 0.168 | 0.086 |
| test_shift | 0.114 | 0.134 | 0.077 |

**H2: gates fitted on cal (debiased+T), auto_err / automation**

| ε | policy | test_in | test_shift |
|---|---|---|---|
| 0.05 | fitted | 0.040 / 0.207 | 0.060 / 0.235 |
| 0.05 | per_choice | 0.030 / 0.203 | 0.048 / 0.223 |
| 0.05 | conformal (LAC) | 0.000 / 0.037 | 0.000 / 0.030 |
| 0.05 | risk | 0.028 / 0.198 | 0.033 / 0.222 |
| 0.1 | fitted | 0.090 / 0.318 | 0.105 / 0.338 |
| 0.1 | per_choice | 0.058 / 0.280 | 0.070 / 0.302 |
| 0.1 | conformal (LAC) | 0.000 / 0.052 | 0.002 / 0.043 |
| 0.1 | risk | 0.058 / 0.268 | 0.047 / 0.283 |

**H2: adaptive risk control on test_shift (site-ordered stream, tail = last 200 steps)**

| ε | policy | auto_err | tail_auto_err | automation |
|---|---|---|---|---|
| 0.05 | risk (static) | 0.033 | 0.020 | 0.222 |
| 0.05 | arisk, r=0 | 0.203 | 0.275 | 0.533 |
| 0.05 | arisk, r=0.05 | 0.035 | 0.055 | 0.190 |
| 0.05 | arisk, r=0.2 | 0.047 | 0.045 | 0.247 |
| 0.1 | risk (static) | 0.047 | 0.030 | 0.283 |
| 0.1 | arisk, r=0 | 0.283 | 0.275 | 0.632 |
| 0.1 | arisk, r=0.05 | 0.155 | 0.015 | 0.387 |
| 0.1 | arisk, r=0.2 | 0.082 | 0.035 | 0.320 |

**H2: spread over 200 calibration draws** (cal re-drawn from cal ∪ test_in, T and gates refit; test_shift fixed; `results/m1/kev_compact_ci.md`)

| ε | policy | test_shift auto_err (2.5–97.5%) | P(auto_err > ε) | automation |
|---|---|---|---|---|
| 0.05 | fitted | 0.065 (0.057–0.072) | 1.000 | 0.245 |
| 0.05 | per_choice | 0.049 (0.042–0.057) | 0.350 | 0.228 |
| 0.05 | risk | 0.037 (0.027–0.045) | 0.000 | 0.226 |
| 0.1 | fitted | 0.109 (0.090–0.125) | 0.875 | 0.347 |
| 0.1 | per_choice | 0.074 (0.065–0.088) | 0.000 | 0.305 |
| 0.1 | risk | 0.054 (0.040–0.070) | 0.000 | 0.294 |

**H3: supervisor (gpt-oss:20b) on escalated rows, full option list vs LAC set (ε=0.1), paired re-run**

| cond | n | acc | abstain | unparsed | acc_given_answer | acc_gold_in_set | prompt_tok | completion_tok | latency_ms |
|---|---|---|---|---|---|---|---|---|---|
| full | 95 | 0.326 | 0.053 | 0.368 | 0.564 | 0.342 | 1700 | 231 | 9543 |
| set | 95 | 0.189 | 0.063 | 0.579 | 0.529 | 0.228 | 1371 | 238 | 8049 |

gold_in_set (LAC coverage on the H3 sample): 0.832; paired n=95 (action_prediction 45, element_grounding 50). Ids answered in both conditions (n=26): full 0.654, set 0.500; discordant 5 / 1, exact McNemar p=0.219.

**Ablation: compact vs full context (shared test_in ids, debiased)**

| mode | n | acc | ECE | Brier | latency_ms | state_chars | n_perm |
|---|---|---|---|---|---|---|---|
| compact | 60 | 0.450 | 0.177 | 0.734 | 5842 | 3646 | 4.233 |
| full | 60 | 0.417 | 0.189 | 0.744 | 23963 | 13141 | 3.000 |

**What held**

- **H1: held, via temperature scaling only.** debiased+T vs first_run ECE: test_in 0.167 → 0.086, test_shift 0.114 → 0.077. Debiasing alone does not help: test_shift ECE 0.114 → 0.134, Brier up on every split.
- **H2: held for risk control, not for LAC.** risk@ε meets ε on held-out sites (0.033 at ε=0.05, 0.047 at ε=0.1). The fitted threshold exceeds ε there (0.060, 0.105); fixed@0.55 gives 0.080, margin@0.1 0.293. Per-choice thresholds also stayed within ε in this run (0.048, 0.070), without a guarantee. LAC sets keep auto_err ≤ 0.002 but automate only 0.030–0.052. Over 200 calibration draws, risk exceeds ε on held-out sites in none (mean 0.037 / 0.054); fitted exceeds it in 200/200 at ε=0.05 and 175/200 at ε=0.1; per-choice in 70/200 at ε=0.05. The control is marginal: per task type, element grounding sits at 0.059 at ε=0.05 on held-out sites (over ε in 70% of draws) and action prediction at 0.015. One gate per type controls each type on calibration sites (action prediction 0.051 / 0.098) but not on held-out sites (0.074 / 0.118, over ε in 100% / 89% of draws): the marginal gate's held-out margin came from pooling. Adaptive risk with one gate per type (audit rate 0.2) brings each type back to ε on held-out streams (action prediction 0.054, grounding 0.045, all 0.049 at ε=0.05), as ACI tracks ε on average per gate (`results/m1/kev_compact_groups.md`).
- **H2 adaptive: needs audits.** arisk at r=0 drifts to α=0.5 (auto_err 0.203 / 0.283). r=0.05 overshoots in this stream (tail 0.055 at ε=0.05; 0.155 overall at ε=0.1), though over 200 redrawn streams it averages 0.056 / 0.091 with a wide spread. r=0.2 stays within ε at both and automates more than static risk (0.247 vs 0.222; 0.320 vs 0.283). ACI on coverage is identical at every audit rate. Over 200 redrawn streams (site order and audits), r=0.2 averages 0.048 / 0.094 with tails tracking ε, but exceeds ε in 40% / 35% of streams: ACI controls the long-run average, not each stream.
- **H3: inconclusive with this supervisor.** Paired re-run (n=95): the set condition is less accurate (0.189 vs 0.326) and cheaper (1371 vs 1700 prompt tokens, 8.0 vs 9.5 s), but 0.368 (full) and 0.579 (set) of replies have no parseable id. On the 26 ids both conditions answered: 0.654 vs 0.500, McNemar p=0.219. Next run: a stronger supervisor with JSON-constrained answers (`--json`).
- **Safety: the keyword critical-action detector has low recall.** On WebGuard (4,956 labelled actions, 193 sites; `results/webguard/detector.md`) it flags 0.162 of HIGH-risk actions from the element label it sees live (0.236 with the page URL) and 0.026 of SAFE ones (0.058). Misses are dominated by "submit", "sign up", "save", "download" and non-English labels; the spec's Kev `noul` detector is not built yet.
- **H4: not run.** The live harness runs in a browser (scripted models); runs with Kev and a supervisor are M2.

**Caveats**

- Zero-shot Kev-4B is weak here (first-run accuracy 0.453–0.485), which caps automation: no gate within ε=0.05 on test_shift automates more than 0.247.
- WebJev labels are LLM-generated and execution-verified; no human-label check yet.
- One executor, one plan. Uncertainty comes from re-drawing cal and a site-cluster bootstrap over the collected rows, not from re-collected data or other seeds of the plan.
- Steps with ≤64 options only (drops ~10% of grounding instances).
- test_in shares sites with cal; only test_shift measures site shift.
- Risk control holds in expectation over calibration draws; q̂ is picked to maximise automation on cal (slight optimism).
- Ablation: n=60, debiased without T, mismatched permutations (mean n_perm 4.233 vs 3.000).
- H3: gpt-oss:20b at "Reasoning: low", max_tokens 1024, weaker than the spec's supervisor; 5 of 100 sampled rows lost to Ollama errors; answered rows 55 (full) / 34 (set).
- The temperature T and the gates are fitted on the same cal rows; the cal rows of `debiased+T` are in-sample, so the conformal and risk-control finite-sample statements are approximate (T is data-dependent).

## Reproduce

```bash
uv sync --group dev && uv run pytest -q

# data
uvx --from huggingface_hub hf download Lexmount/WebJev --repo-type dataset --local-dir data/webjev

# executor: Kev-4B on :8009 (MLX bf16 on Apple Silicon)
git clone https://github.com/jaredpalmer/kev third_party/kev
(cd third_party/kev && uv sync --extra serve && uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b --port 8009)

# plan: 30% of sites held out
uv run python scripts/make_plan.py --root data/webjev --n-cal 600 --n-test-in 400 --n-test-shift 600 --max-options 64 --seed 0 --out runs/plan.json

# collect (resumable): compact, 5 permutations (3 above 16 options); full-context ablation on 60 test_in rows
uv run python scripts/offline_collect.py --plan runs/plan.json --backend kev --mode compact --out runs/kev_compact.jsonl
uv run python scripts/offline_collect.py --plan runs/plan.json --backend kev --mode full --n-perm 3 --splits test_in --limit 60 --out runs/kev_full_test_in.jsonl

# H3 supervisor: gpt-oss:20b on Ollama (:11434)
uv run python scripts/offline_supervisor.py --collect runs/kev_compact.jsonl --eps 0.1 --n 100 --reasoning low --max-tokens 1024 --out runs/h3_0.1.jsonl

# tables + risk-coverage plot
uv run python scripts/offline_analyze.py --collect runs/kev_compact.jsonl --ablation runs/kev_full_test_in.jsonl --h3 runs/h3_0.1.jsonl --out results/m1/kev_compact.md

# uncertainty: 200 calibration draws, site-cluster bootstrap, 200 adaptive streams (CPU, ~2 min)
uv run python scripts/offline_ci.py --collect runs/kev_compact.jsonl --seeds 200 --boot 2000 --out results/m1/kev_compact_ci.md

# critical-detector recall on WebGuard (CC-BY-NC-4.0: evaluation only)
uvx --from huggingface_hub hf download boyuanzheng010/webguard --repo-type dataset --local-dir data/webguard_alt/webguard
uv run python scripts/eval_webguard.py --out results/webguard/detector.md
```

The committed rows reproduce every table without a model: `--collect artifacts/m1/kev_compact.jsonl --ablation artifacts/m1/kev_full_test_in.jsonl --h3 artifacts/m1/h3_0.1b.jsonl`.

Windows 11: setup, tests, data, plan (byte-for-byte identical), fine-tune data, tables, uncertainty and WebGuard steps run as written; collection and H3 need the model servers and were not run there. Set `HF_HUB_DISABLE_SYMLINKS=1` if Hugging Face downloads fail with WinError 1314, and `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` if a stale stored token makes a public repo 404. Kev's README lists L40S/H100 for Kev-4B on CUDA and a 4 GB GPU for Kev-0.8B; on Windows its lockfile installs CPU torch, and on a 4 GB laptop GPU with 16 GB of RAM, Kev-0.8B's first request grew the pagefile by ~4 GB.

**Kev-4B fine-tune on Modal** (H100, ~$1–3): `uvx --from modal==1.5.5 modal run scripts/modal_train_kev.py --max-steps 5` (smoke), then `--name webjev-v1`; it reuses Kev's image and passes `--max_state 7552`, which Kev's own fine-tune skill does not (its 384-token default drops ~94% of WebJev records).

**Live (MiniWoB++)**: `uv sync --group dev --extra live && uv run playwright install chromium`, serve `miniwob-plusplus/miniwob/html` on :8765, `MINIWOB_URL=http://127.0.0.1:8765/miniwob/`, then e.g. `uv run python scripts/run_miniwob.py --task login-user --backend kev --policy arisk:0.05 --fit artifacts/m1/kev_compact.jsonl --audit-rate 0.2 --supervisor-url ... --supervisor-json`. `SURETY_LIVE=1 uv run pytest tests/integration` runs the browser tests with scripted models.

**H4 (MiniWoB++)**: `uv run python scripts/run_h4.py --tasks @configs/miniwob_h4.txt --seeds 0,1,2 --fit artifacts/m1/kev_compact.jsonl --backend kev --supervisor-url ... --supervisor-json --shadow-rate 0.2 --out runs/h4/miniwob.jsonl` runs the six systems (supervisor, executor, fixed, escalate_on_none, risk, arisk) on 68 text-centric tasks, resumably, and writes the H4 table next to it; `--shadow-rate` asks the supervisor about a sample of acted steps (evaluation calls, reported apart from each system's cost) so the table estimates live autonomous error per system. `configs/miniwob_safety.txt` lists tasks that should end in a safety stop. Both live scripts share the supervisor flags `--supervisor-max-tokens` (default 1,024: reasoning models spend tokens before answering), `--supervisor-reasoning`, `--supervisor-api-key-env`, and `--supervisor-no-temperature` / `--supervisor-token-param max_completion_tokens` for OpenAI reasoning models. A run resumes from its `--out` file (α and audit draws included) and refuses changed settings.

## Status / next (M2)

- M1 offline study on Kev-4B: done, with calibration-draw and site-bootstrap uncertainty.
- Live harness: runs in Chromium on MiniWoB++ with scripted models (login-user stops at the password field with `stop:credentials`; click-test reward 1.0). The gate is fitted offline (`--fit`, scored with the fit's 5/3 orderings), calibrated live, and updated online from escalations and audits; `run_h4.py` keeps one gate per system across its task stream, while `run_miniwob.py` plays a single episode. Every MiniWoB++ task opens without error; clickable spans/divs/icons are now options (tasks with no option 27 → 4, `results/miniwob_observe.md`). H4 runner ready (`scripts/run_h4.py`); no run with real models yet.
- Kev UI fine-tune: data built (`scripts/make_kev_train.py`: 8,000 train / 500 held-out on 27 whole sites, disjoint from every evaluation row); Modal launcher ready (`scripts/modal_train_kev.py`), not run.
- H3 with a stronger supervisor: `scripts/offline_supervisor.py --json --api-key-env <VAR> --base-url <endpoint> --model <model>` (`--no-temperature --token-param max_completion_tokens` for OpenAI reasoning models).
- Critical detection: add the spec's Kev `noul` check; candidate keyword families and their WebGuard trade-offs are in `results/webguard/detector.md`.
- Clef-flash as a second executor (Apple Silicon); Mind2Web (human labels); then WebArena-Verified hard subset and H4.

## License

Apache-2.0 (`LICENSE`).

Builds on [Kev](https://github.com/jaredpalmer/kev), [WebJev](https://huggingface.co/datasets/Lexmount/WebJev) (Lexmount), [Clef](https://huggingface.co/mlx-community/clef-flash-4bit), conformal risk control (Angelopoulos et al.) and adaptive conformal inference (Gibbs & Candès).
