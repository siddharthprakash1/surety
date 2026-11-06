# Surety — design spec

Date: 2026-11-06. Status: approved in conversation (parts 1–3); part 4 decided by the implementer.

## Goal

An open-source computer-use-agent harness where a small local decision model takes routine per-step actions and a reasoning LLM supervises, with **escalation that carries an error guarantee**: autonomous wrong actions ≤ ε per step, maintained online as websites change.

Deliverables: the harness, an offline calibration study (M1), end-to-end results (M2), tech report + release (M3).

## Non-goals

- Beating accuracy leaderboards (OSWorld-Verified ≈ 90 %, saturated).
- Training a new decision model (only if the fine-tune trigger fires).
- Desktop apps. Web only (DOM / accessibility tree).
- A hosted product.

## Prior art and the gap

| Work | What it does | Missing |
|---|---|---|
| Jev-Mobile (2609.30186) | VLM plans, Jev executes on Android; returns control on DONE/BLOCKED | no confidence gate, no calibration, no code |
| JevAgentHarness (2610.21412) | per-choice thresholds fitted on own trajectories | no guarantee, no web/GUI tasks |
| Step-level cascade (2604.27151) | BERT monitors trigger escalation | hand-tuned, no guarantee |
| CORA (2604.09155), KnowNo | conformal act/abstain | mobile harm / robotics only |
| arc-cua, browser repos, Vercel fallback | fixed thresholds 0.55–0.7 | assume calibration nobody measured |

Gap: calibrated sets over enumerated page elements + per-step conformal gate + supervisor choosing *within the set* + adaptive recalibration under site shift, evaluated on web tasks.

## Architecture

```
task ─► Supervisor.plan ─► subgoals + text slots
   step ▼
   Env ─► Page ─► Compiler ─► DecisionState (subgoal + shortlist, ≤ ~1k tokens)
                                   │
                      Executor (Kev / Clef / Jev / LLM-logprob) ─► probabilities
                                   │
                      Gate ─► act | escalate(reason, candidate set)
                       │                        │
                    Env.act         Supervisor.choose / verify ─► Env.act
                                   │
                      Trace ─► JSONL (state, options, probs, gate, supervisor, latency, tokens)
```

Units (`src/surety/`):

| Unit | Responsibility | Interface |
|---|---|---|
| `env` | normalize a page; execute actions | `Page{url,title,text,elements[Element{id,role,name,value,bbox?}]}`, `Env.observe()`, `Env.act(Action)` |
| `compile` | Page → shortlist → `DecisionState`; richer supervisor view | `compile(page, subgoal, history) -> DecisionState{state_text, options[Option{id,label}], critical_hints}` |
| `decide` | one interface over Jev's `choice`/`noul`/`score` | `Backend.ask(state_text, questions) -> Answers`; `DebiasedBackend` (option-order permutations, noul polarity flip); `Recalibrator` (temperature per question type) |
| `gate` | probabilities → `Act(option)` or `Escalate(reason, set)` | `Policy.decide(probs) `; policies: `FixedThreshold`, `Margin`, `PerChoiceThreshold`, `Conformal(ε)`, `AdaptiveConformal(ε, γ, audit_rate)`; `CriticalOverride` always wraps |
| `supervise` | plan / choose / verify / replan over OpenAI-compatible chat | `Supervisor.plan(task, page)`, `.choose(state, options)`, `.verify(action, state)`, `.replan(...)` |
| `loop` | step loop, budgets, stuck detection | `run(task, env, executor, gate, supervisor, trace) -> Episode` |
| `trace` | append-only JSONL per decision | `Trace.record(Decision)` |
| `calib` | metrics and fitting | ECE, Brier, coverage, set size, automation rate, risk–coverage; temperature fit; conformal quantile |
| `eval` | offline replay and live runners | `webjev`, `mind2web` adapters; `browsergym` runner |

Invariants: only the supervisor writes free text; the executor only picks from enumerated options. All envs produce the same `Page`, so offline calibration transfers to live runs and the gap is measured, not assumed.

## Escalation design

Per decision:

1. **Debias.** Same `choice` question in k shuffled option orders (Kev: `/v1/systemone/permute`), `noul` asked both polarities; average.
2. **Recalibrate.** Temperature per question type fitted on logged UI decisions (Kev's shipped temperatures were fitted on text classification).
3. **Conformal set.** Split conformal on held-out labeled steps (score = 1 − p(option)); set C = {o : score ≤ q̂₁₋ε}. |C|=1 → act (guarantee applies). |C|≥2 → supervisor chooses within C. `none_of_these` ∈ C or C=∅ → supervisor sees full view.
4. **Hard overrides.** Critical actions (pay, buy, send, delete, submit-with-side-effects, account changes) → supervisor verify; money/credentials → stop, hand to human. Detection = rules + debiased `noul`, tuned for recall. No page change / repeated action → replan.

Online: adaptive conformal inference updates q̂ after every labeled step (`α_{t+1} = α_t + γ(ε − err_t)`). Labels come from escalations (biased toward hard steps) plus a random audit of confident steps at rate r (weight 1/r). Guarantee is relative to the supervisor; ground truth from offline labels and task success.

Baselines on identical code: supervisor-only, executor-only, fixed thresholds (0.55, 0.7), margin (top-2 gap < 0.1), per-choice fitted thresholds (JevAgentHarness-style), executor `confidence` field, Jev-Mobile rule (escalate on done/blocked only).

## Evaluation

### Offline (M1)

Data:
- **WebJev** (Lexmount, Apache-2.0): 64,122 decisions, 3,858 tasks, 1,443 sites; `action_prediction` (3–13 options) and `element_grounding` (≤137 options). Fields: `context`, `question`, `options`, `gold`, `website`, `num_tokens`. Site-level splits: calibration sites vs held-out sites.
- **Mind2Web** train (human labels, candidates provided): secondary, for comparability; held-out website/domain splits built from train. Deferred to M2 if time-bound.

Executors: Kev-4B (default), Clef-flash 9B (MLX 4-bit, image-capable, Jev API). Hosted Jev optional.

Supervisor answers precomputed on a stratified sample so every gate policy and ε is simulated offline.

Hypotheses (fixed before runs; failures published):
- **H1** Debias + recalibrate lowers ECE on every split, including held-out sites.
- **H2 (headline)** On held-out sites, fitted/fixed/margin thresholds miss their error target; conformal + ACI with audits stays within ε.
- **H3** Supervisor given the conformal set (not the full option list) uses fewer tokens at equal step accuracy.
- **H4** End to end: within 2 pts of supervisor-only success at ≤ 35 % of its calls.

Metrics: accuracy, ECE (15 bins), Brier, coverage vs 1−ε, mean set size, automation rate (|C|=1 share), risk–coverage curve, confident-error rate (p≥0.9 and wrong), supervisor calls/tokens, latency.

Fine-tune trigger: zero-shot automation < 50 % at ε=0.05 on calibration sites → UI LoRA on WebJev train split (Kev training recipe).

### Live (M2)

BrowserGym pinned to the commit with `webarena_verified_hard` (2026-10-27). MiniWoB++ first; WebArena-Verified hard subset (258 tasks) if x86 images run under emulation on the Mac (one-day test). Six systems on identical tasks/seeds; full set for 3 main systems, fixed 100-task sample for ablations. Report tokens and cost at Jev list price and a frontier list price.

## Safety

- Executor never generates text; page text is data. Supervisor text generation limited to slot values and plans.
- Critical-action override cannot be disabled by any policy; money/credentials always stop.
- Detector recall measured on WebGuard (4,939 labeled actions).
- Budgets: max steps, wall time, escalations per task.
- Traces never store credentials; env redacts password fields.

## Testing

- Unit: `MockBackend` (deterministic probabilities), `MockSupervisor`; gate policies on synthetic data with known truth (coverage ≥ 1−ε within binomial tolerance, fixed seed); calib metrics against hand-computed values; compile on fixture pages; critical detector on fixtures.
- Integration (marked): Playwright on a static HTML fixture; MiniWoB++ 3 tasks with mocks; real-model smoke behind `SURETY_LIVE=1`.
- Experiments are scripts under `scripts/`, outputs under `runs/` (gitignored), results tables regenerated by script.

## Repo

`surety/` — uv, Python 3.12, Apache-2.0. `src/surety/{env,compile,decide,gate,supervise,loop,trace,calib,eval}`, `tests/`, `scripts/`, `configs/`, `docs/superpowers/{specs,plans}`, `third_party/kev` (git clone, gitignored), `data/` and `runs/` gitignored. PyPI name `surety-cua`.

Model serving (all local, MLX): Kev `python -m kev.serve --run jaredpalmer/kev-4b --port 8009`; Clef-flash via bundled `clef_mlx.py` wrapped in `/v1/systemone`; supervisor `mlx_lm.server --model mlx-community/Qwen3.6-35B-A3B-4bit`.

## Milestones

| | Date | Deliverable |
|---|---|---|
| M1 | 2026-11-29 | Offline calibration study + blog post |
| M2 | 2027-01-03 | End-to-end results table |
| M3 | 2027-01-17 | Tech report, release, demo video |

## Risks

- Kev-4B trained on ≤384-token states; WebJev contexts reach 16k. Mitigation: shortlist/truncate to ~1k tokens; measure accuracy vs length; Clef as second executor.
- WebJev labels are LLM-generated (execution-verified). Mitigation: Mind2Web human labels as secondary check.
- WebArena images are amd64-only. Mitigation: emulation test early; fall back to MiniWoB++ + Online-Mind2Web sample.
- 48 GB RAM, GPU limit ≈ 75 %. Never co-load Clef + Kev + supervisor; offline study runs one executor at a time.
