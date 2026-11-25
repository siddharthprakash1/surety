# Surety

Surety is a web computer-use agent harness. A small local decision model picks each step's action from an enumerated list, and a reasoning LLM supervises. A conformal gate decides who acts, keeping wrong autonomous actions at or below a target ε per step, and adapts online from escalations and random audits as sites change.

## Design

Per step, the executor (Kev-4B, served locally) scores the options in several shuffled orders (debias), the averaged probabilities are temperature-scaled on calibration data (recalibrate), and a gate either acts on one option or escalates to the supervisor (any OpenAI-compatible chat model) with a candidate set. Gate families on identical code: fixed, margin, fitted and per-choice probability thresholds; split-conformal LAC sets (coverage control); conformal risk control on the autonomous-error loss 1{|C|=1 ∧ gold∉C}; adaptive versions of the last two that update α online from escalated steps plus a random audit of acted steps (weight 1/r); and gates on a learned error score (`src/surety/gate/learned.py`: a cross-fitted logistic model of P(top option wrong) from five features of the probability vector, thresholded by risk control, by Learn-then-Test for P(auto_err > ε) ≤ δ, or adaptively with doubly-robust audit feedback). A critical-action override (pay, send, delete, account changes) wraps every policy: payment stops the episode, the other categories escalate with the full option list; typing into or selecting in password, card or other credential fields stops the episode. Spec: `docs/superpowers/specs/2026-11-06-surety-design.md`.

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

**H2: learned error score, 200 calibration draws** (2026-11-24; `results/m1/kev_compact_learned.md`). Held-out sites (test_shift), mean auto_err (P(auto_err > ε)) / automation:

| ε | risk (above) | learned_pac (LTT, δ = 0.1) | learned (CRC) |
|---|---|---|---|
| 0.05 | 0.037 (0.000) / 0.226 | 0.038 (0.005) / 0.274 | 0.046 (0.130) / 0.302 |
| 0.1 | 0.054 (0.000) / 0.294 | 0.078 (0.020) / 0.379 | 0.095 (0.260) / 0.411 |

Adaptive on 200 test_shift streams (audit rate 0.2): arisk 0.048 (over ε in 40 % of streams) / 0.236 at ε = 0.05 and 0.094 (35 %) / 0.348 at ε = 0.1; alearned 0.036 (9 %) / 0.267 and 0.075 (4 %) / 0.372.

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
- **Learned error score: more automation at the same error.** Risk control ranks steps by p_top; a five-feature logistic estimate of P(top option wrong), cross-fitted on cal, ranks executor errors on held-out sites at AUROC 0.84 vs 0.70. Thresholded by Learn-then-Test (δ = 0.1), it matches risk control's realized error on held-out sites at ε = 0.05 (0.038 vs 0.037, over ε in 1 of 200 draws) while automating 0.274 instead of 0.226, and 0.379 instead of 0.294 at ε = 0.1 (0.078 vs 0.054); across ε = 0.02–0.20 it automates more at every level (0.551 vs 0.432 at ε = 0.2). It also balances the task types that marginal risk control left uneven: at ε = 0.05, grounding 0.039 (risk: 0.059, over ε in 70 % of draws) and action prediction 0.038 at twice the automation (0.160 vs 0.083). The CRC version sits on ε (0.046 / 0.095) for the most automation (0.302 / 0.411). Fitting T and the model on 400 separate rows and calibrating on the 600 cal rows (exact validity) gives the same numbers. Adaptive, with a smaller step (γ = 0.005), a 0.8·ε target and doubly-robust audit feedback, the learned gate exceeds ε in 9 % / 4 % of streams instead of arisk's 40 % / 35 %, at more automation (0.267 vs 0.236; 0.372 vs 0.348); those three settings were picked on the same streams. One driver: on action-prediction steps zero-shot Kev's top pick is BLOCKED 30 % of the time and wrong on 239 of those 242 steps, which the terminal-option feature catches.
- **Option-prior correction: more accuracy, not yet safe automation on new sites** (2026-11-25; `results/m1/kev_compact_prior.md`). Eight per-action log-prior offsets fitted on cal (BLOCKED −2.85, CLICK +2.67, ...) lift top-1 accuracy on held-out sites from 0.458 to 0.588 (action prediction 0.310 → 0.570). On calibration sites the PAC gate then automates 0.306 / 0.453 at ε = 0.05 / 0.1 (from 0.273 / 0.388) within ε, but on held-out sites the gates overshoot (learned_pac 0.052 and 0.116, over ε in 49 % / 92 % of draws), all from action-prediction steps whose errors rank worse there; adaptive gates hold ε by giving most of the gain back (alearned 0.266 / 0.406 automation, vs 0.267 / 0.372 without). Not adopted by default; it points at the fine-tune and at calibration data from more sites.
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
- Learned gates: the five features were chosen after an exploratory look at test AUROC (9 candidates), and alearned's γ, margin and feedback rule after 24 simulated configurations on the test_shift streams; both are in-sample choices to confirm on new data. P(auto_err > ε) in the tables is the share of draws whose *realized* rate on the test rows exceeds ε; Learn-then-Test bounds the probability that the *true* risk does, so finite-test noise alone pushes the realized share above δ (0.125 / 0.140 on test_in).

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

# learned error score vs p_top: error ranking, 200 calibration draws, 200 adaptive streams, ε sweep (CPU, ~15 min)
uv run python scripts/offline_learned.py --collect runs/kev_compact.jsonl --out results/m1/kev_compact_learned.md

# option-prior correction of the executor through the same draws and streams (CPU, ~10 min)
uv run python scripts/offline_prior.py --collect runs/kev_compact.jsonl --out results/m1/kev_compact_prior.md

# critical-detector recall on WebGuard (CC-BY-NC-4.0: evaluation only)
uvx --from huggingface_hub hf download boyuanzheng010/webguard --repo-type dataset --local-dir data/webguard_alt/webguard
uv run python scripts/eval_webguard.py --out results/webguard/detector.md
```

The committed rows reproduce every table without a model: `--collect artifacts/m1/kev_compact.jsonl --ablation artifacts/m1/kev_full_test_in.jsonl --h3 artifacts/m1/h3_0.1b.jsonl`.

Windows 11: setup, tests, data, plan (byte-for-byte identical), fine-tune data, tables, uncertainty and WebGuard steps run as written; collection and H3 need the model servers and were not run there. Set `HF_HUB_DISABLE_SYMLINKS=1` if Hugging Face downloads fail with WinError 1314, and `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` if a stale stored token makes a public repo 404. Kev's README lists L40S/H100 for Kev-4B on CUDA and a 4 GB GPU for Kev-0.8B; on Windows its lockfile installs CPU torch, and on a 4 GB laptop GPU with 16 GB of RAM, Kev-0.8B's first request grew the pagefile by ~4 GB.

**Kev-4B fine-tune on Modal (M2)**: a fine-tuned Kev must never train on a calibration or test site, so it has its own plan with three disjoint site groups (721 training, 361 calibration, 361 test), each split spread over its sites (at most 4 steps a site; M1's top 10 sites hold 36–41 % of each split):

```bash
uv run python scripts/make_plan.py --design spread --cal-frac 0.25 --test-frac 0.25 --n-cal 600 --n-test-in 400 --n-test-shift 600 --max-options 64 --seed 0 --out artifacts/m2/plan.json
uv run python scripts/make_kev_train.py --plan artifacts/m2/plan.json   # runs/kev_ft: 8,000 train / 500 held-out steps, training sites only
uvx modal setup                                                         # once; H100s need a payment method
uvx --from modal==1.5.5 modal run scripts/modal_train_kev.py --max-steps 5   # smoke
uvx --from modal==1.5.5 modal run scripts/modal_train_kev.py --name webjev-v1   # ~1–2 H100-hours
# score in one container (kev.serve + offline_collect; the checkpoint stays on Modal): parity with M1's rows first
uvx --from modal==1.5.5 modal run scripts/modal_score_kev.py --run zero-shot --plan artifacts/m1/plan.json --name m1-parity --limit 100
uvx --from modal==1.5.5 modal run scripts/modal_score_kev.py --run zero-shot --plan artifacts/m2/plan.json --name m2-zero-shot
uvx --from modal==1.5.5 modal run scripts/modal_score_kev.py --run webjev-v1 --plan artifacts/m2/plan.json --name m2-webjev-v1
uvx --from modal==1.5.5 modal volume get surety-kev-ft scores/m1-parity runs/scores/m1-parity   # likewise each --name
uv run python scripts/score_parity.py artifacts/m1/kev_compact.jsonl runs/scores/m1-parity/rows.jsonl
```

The training launcher reuses Kev's image and passes `--max_state 7552`, which Kev's own fine-tune skill does not (its 384-token default drops ~94% of WebJev records). A re-score of 1,600 steps should take well under an hour on an H100 (estimate; M1's took 1.8 h on Apple Silicon); the zero-shot re-score on M2 is the like-for-like baseline, since H100 bf16 numerics differ from M1's MLX rows.

**Live (MiniWoB++)**: `uv sync --group dev --extra live && uv run playwright install chromium`, serve `miniwob-plusplus/miniwob/html` on :8765, `MINIWOB_URL=http://127.0.0.1:8765/miniwob/`, then e.g. `uv run python scripts/run_miniwob.py --task login-user --backend kev --policy arisk:0.05 --fit artifacts/m1/kev_compact.jsonl --audit-rate 0.2 --supervisor-url ... --supervisor-json`. `SURETY_LIVE=1 uv run pytest tests/integration` runs the browser tests with scripted models.

**H4 (MiniWoB++)**: `uv run python scripts/run_h4.py --tasks @configs/miniwob_h4.txt --seeds 0,1,2 --fit artifacts/m1/kev_compact.jsonl --backend kev --supervisor-url ... --supervisor-json --shadow-rate 0.2 --out runs/h4/miniwob.jsonl` runs the six systems (supervisor, executor, fixed, escalate_on_none, risk, arisk) on 68 text-centric tasks, resumably, and writes the H4 table next to it; `--shadow-rate` asks the supervisor about a sample of acted steps (evaluation calls, reported apart from each system's cost) so the table estimates live autonomous error per system. `configs/miniwob_safety.txt` lists tasks that should end in a safety stop. Add the learned-score gates with `--systems supervisor,executor,fixed,escalate_on_none,risk,arisk,learned,alearned` (`learned` = learned_pac at `--eps`; `alearned` is audited like arisk); `run_miniwob.py --policy learned_pac:0.05` plays one episode with it. Both live scripts share the supervisor flags `--supervisor-max-tokens` (default 1,024: reasoning models spend tokens before answering), `--supervisor-reasoning`, `--supervisor-api-key-env`, and `--supervisor-no-temperature` / `--supervisor-token-param max_completion_tokens` for OpenAI reasoning models. A run resumes from its `--out` file (α, audit draws and audit counts included) and refuses changed settings. Answered audits weigh 1/(r·q̂), with q̂ the stream's share of audits the supervisor answered, so unparsed replies do not bias the adaptive gates (still prefer `--supervisor-json`); an episode the browser could not run is retried without touching the gate.

## Status / next (M2)

- M1 offline study on Kev-4B: done, with calibration-draw and site-bootstrap uncertainty.
- Learned-score gates (2026-11-24): `learned`, `learned_pac`, `alearned` in `--policy` / `--fit` specs and as opt-in H4 systems; offline results above; not yet run live with models. Research behind them and the next candidates: `docs/research/2026-11-24-improvements.md`.
- Live harness: runs in Chromium on MiniWoB++ with scripted models (login-user stops at the password field with `stop:credentials`; click-test reward 1.0). The gate is fitted offline (`--fit`, scored with the fit's 5/3 orderings), calibrated live, and updated online from escalations and audits; `run_h4.py` keeps one gate per system across its task stream, while `run_miniwob.py` plays a single episode. Every MiniWoB++ task opens without error; clickable spans/divs/icons are now options (tasks with no option 27 → 4, `results/miniwob_observe.md`). H4 runner ready (`scripts/run_h4.py`); no run with real models yet.
- Kev UI fine-tune (M2): plan, data and Modal jobs ready, not run (needs a Modal account). `artifacts/m2/plan.json` keeps training, calibration and test sites disjoint; the training data (8,000 / 500 steps from 615 / 31 training sites, `artifacts/m2/kev_ft_stats.json`) has BLOCKED as the answer on 1.2 % of action steps, against zero-shot Kev's 30 % BLOCKED-first rate. `scripts/modal_score_kev.py` is tested against a stand-in server, not on Modal.
- H3 with a stronger supervisor: `scripts/offline_supervisor.py --json --api-key-env <VAR> --base-url <endpoint> --model <model>` (`--no-temperature --token-param max_completion_tokens` for OpenAI reasoning models).
- Critical detection: add the spec's Kev `noul` check; candidate keyword families and their WebGuard trade-offs are in `results/webguard/detector.md`.
- Clef-flash as a second executor (Apple Silicon); Mind2Web (human labels); then WebArena-Verified hard subset and H4.

## License

Apache-2.0 (`LICENSE`).

Builds on [Kev](https://github.com/jaredpalmer/kev), [WebJev](https://huggingface.co/datasets/Lexmount/WebJev) (Lexmount), [Clef](https://huggingface.co/mlx-community/clef-flash-4bit), conformal risk control (Angelopoulos et al.) and adaptive conformal inference (Gibbs & Candès).
