# Surety

Surety is a web computer-use agent harness. A small local decision model picks each step's action from an enumerated list, and a reasoning LLM supervises. A conformal gate decides who acts, keeping wrong autonomous actions at or below a target ε per step, and adapts online from escalations and random audits as sites change.

## Design

Per step, the executor (Kev-4B, served locally) scores the options in several shuffled orders (debias), the averaged probabilities are temperature-scaled on calibration data (recalibrate), and a gate either acts on one option or escalates to the supervisor (any OpenAI-compatible chat model) with a candidate set. Gate families on identical code: fixed, margin, fitted and per-choice probability thresholds; split-conformal LAC sets (coverage control); conformal risk control on the autonomous-error loss 1{|C|=1 ∧ gold∉C}; adaptive versions of the last two that update α online from escalated steps plus a random audit of acted steps (weight 1/r). A critical-action override (pay, send, delete, account changes) wraps every policy; typing into password or other credential fields stops the episode. Spec: `docs/superpowers/specs/2026-11-06-surety-design.md`.

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
| 0.05 | conformal (LAC) | 0.000 / 0.037 | 0.000 / 0.028 |
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
| 0.1 | arisk, r=0.05 | 0.125 | 0.040 | 0.320 |
| 0.1 | arisk, r=0.2 | 0.082 | 0.035 | 0.320 |

**H3: supervisor (gpt-oss:20b) on escalated rows, full option list vs LAC set (ε=0.1)**

| cond | n | acc | abstain | unparsed | acc_given_answer | prompt_tok | completion_tok | latency_ms |
|---|---|---|---|---|---|---|---|---|
| full | 97 | 0.268 | 0.041 | 0.464 | 0.542 | 1988.309 | 253.155 | 19893.480 |
| set | 103 | 0.223 | 0.097 | 0.563 | 0.657 | 1494.845 | 239.981 | 14965.849 |

gold_in_set (LAC coverage on escalated rows): 0.816.

**Ablation: compact vs full context (shared test_in ids, debiased)**

| mode | n | acc | ECE | Brier | latency_ms | state_chars | n_perm |
|---|---|---|---|---|---|---|---|
| compact | 60 | 0.450 | 0.177 | 0.734 | 5842 | 3646 | 4.233 |
| full | 60 | 0.417 | 0.189 | 0.744 | 23963 | 13141 | 3.000 |

**What held**

- **H1: held, via temperature scaling only.** debiased+T vs first_run ECE: test_in 0.167 → 0.086, test_shift 0.114 → 0.077. Debiasing alone does not help: test_shift ECE 0.114 → 0.134, Brier up on every split.
- **H2: held for risk control, not for LAC.** risk@ε meets ε on held-out sites (0.033 at ε=0.05, 0.047 at ε=0.1). The fitted threshold exceeds ε there (0.060, 0.105); fixed@0.55 gives 0.080, margin@0.1 0.293. Per-choice thresholds also stayed within ε in this run (0.048, 0.070), without a guarantee. LAC sets keep auto_err ≤ 0.002 but automate only 0.028–0.052.
- **H2 adaptive: needs audits.** arisk at r=0 drifts to α=0.5 (auto_err 0.203 / 0.283). r=0.05 overshoots (tail 0.055 at ε=0.05; 0.125 overall at ε=0.1). r=0.2 stays within ε at both and automates more than static risk (0.247 vs 0.222; 0.320 vs 0.283). ACI on coverage is identical at every audit rate.
- **H3: inconclusive.** 0.464 (full) and 0.563 (set) of supervisor replies have no parseable id. Among answered rows the set condition is more accurate (0.657 vs 0.542). Averaged over all rows, set prompts are 1495 vs 1988 tokens and 15.0 vs 19.9 s. Raw accuracy is lower with the set (0.223 vs 0.268).
- **H4: not run** (needs the live harness, M2).

**Caveats**

- Zero-shot Kev-4B is weak here (first-run accuracy 0.453–0.485), which caps automation: no gate within ε=0.05 on test_shift automates more than 0.247.
- WebJev labels are LLM-generated and execution-verified; no human-label check yet.
- One executor, one seed (0), one plan; no confidence intervals.
- Steps with ≤64 options only (drops ~10% of grounding instances).
- test_in shares sites with cal; only test_shift measures site shift.
- Risk control holds in expectation over calibration draws; q̂ is picked to maximise automation on cal (slight optimism).
- Ablation: n=60, debiased without T, mismatched permutations (mean n_perm 4.233 vs 3.000).
- H3: gpt-oss:20b at "Reasoning: low", max_tokens 1024, weaker than the spec's supervisor; n differs by condition (97 / 103); answered rows 48 (full) / 35 (set).

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
```

## Status / next (M2)

- M1 offline study on Kev-4B: done.
- Kev UI fine-tune: data built (`scripts/make_kev_train.py`: 8,000 train / 500 held-out on 27 whole sites, disjoint from every evaluation row); training (`scripts/train_kev.sh`, try `--max_steps 5` first) not yet run.
- Clef-flash as a second executor.
- Live runs: MiniWoB++ via `scripts/run_miniwob.py` (harness built, no browser run yet), then WebArena-Verified hard subset; H4.
- Mind2Web (human labels); WebGuard (critical-detector recall).

## License

Apache-2.0 (`LICENSE`).

Builds on [Kev](https://github.com/jaredpalmer/kev), [WebJev](https://huggingface.co/datasets/Lexmount/WebJev) (Lexmount), [Clef](https://huggingface.co/mlx-community/clef-flash-4bit), conformal risk control (Angelopoulos et al.) and adaptive conformal inference (Gibbs & Candès).
