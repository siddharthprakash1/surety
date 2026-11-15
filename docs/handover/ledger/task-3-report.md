# Task 3 report: Gate policies

**Status:** DONE_WITH_CONCERNS
**Branch:** `worktree-agent-a542998465fe6ada9` · **Commit:** `e88e753` feat(gate): threshold, conformal, adaptive, critical override
**Tests:** `uv run pytest -q` → 24 passed (1 existing + 12 gate + 11 critical incl. 7 parametrized)

## Files
- `src/surety/gate/__init__.py`: re-exports all policies plus `GatePolicy`
- `src/surety/gate/policies.py`: `GatePolicy`, `FixedThreshold`, `Margin`, `FittedThreshold`, `PerChoiceThreshold`, `SplitConformal`, `AdaptiveConformal`
- `src/surety/gate/critical.py`: `CriticalDetector`, `CriticalOverride`
- `tests/test_gate.py`, `tests/test_critical.py`

## Behavior pinned beyond the brief
- `name`: `fixed@τ`, `margin@δ`, `fitted@ε`, `per_choice@ε`, `conformal@ε`, `aci@ε`, `<inner>+critical`.
- Non-critical `reason` strings: thresholds `confident`/`low_confidence`; margin `margin`/`low_margin`; conformal `singleton`/`set`/`empty_set`.
- Unfitted defaults are conservative: `FittedThreshold`/`PerChoiceThreshold` τ=∞ (never act); `SplitConformal`/`AdaptiveConformal` q=∞ (full set).
- `_fit_tau` returns ∞ (never act) if no grid τ meets ε. The grid is `np.round(np.arange(0, 1.0001, 0.01), 2)`, rounded to remove float drift such as 0.07000000000000001.
- `Margin` treats p2=0 for a single option, so single-option instances act under every policy (tested).
- `SplitConformal.prediction_set(probs)` is public: the set sorted by prob desc. ACI's `update` uses it to compute `err`.
- ACI `_q`: q=∞ (full set) when `alpha <= alpha_min`; otherwise `np.quantile(scores, 1 - alpha, method="higher")`, the plain empirical quantile without the finite-sample correction (as instructed; matches Gibbs & Candès). `fit` does not reset `alpha`.
- `CriticalOverride`: if an option id has no label, it classifies the id itself instead of skipping the check. Multi-word keywords match any whitespace (`\s+`). Category priority follows the brief's order: payment, destructive, send, account. `classify` searches `f"{label} {url}"`.

## Numbers (synthetic, seed 0, 2k cal / 2k test)
- ε=0.1: q=0.877, coverage 0.8925, mean set size 2.08, automation 27.6%, autonomous error 3.1%. ACI at α=ε gives coverage 0.892.
- Coverage test robustness: seeds 0–19 all land in [0.885, 0.9165], so all are inside [0.88, 0.93].
- FittedThreshold ε=0.1: τ=0.63. Per-choice τ ranges from 0.60 to 0.65.

## Concerns
1. The keyword list is used verbatim. With word boundaries it misses common variants: `\bpay\b` misses "Payment"/"PayPal", so "Proceed to payment" and "Send payment" are not flagged as `stop:payment` ("Send payment" becomes `critical:send`). Other misses: `remove`→"Removed", `log out`→"Logout", `sign out`→"Sign-out". The design says detection is tuned for recall, so consider adding stems/variants (`pay(ment)?`, `log\s*-?out`, …) in a follow-up. The tests do not pin these misses as desired behavior.
2. Because `classify` also searches `url`, a URL that is the current page rather than the option's target would flag every option on e.g. `/checkout`. Callers should pass the option's target URL.
3. The ACI quantile has no finite-sample correction, so at α=ε it differs very slightly from `SplitConformal` (q 0.8766 vs 0.8768 here). If they should match exactly, switch ACI to `ceil((n+1)(1-α))/n`.

## Fix round 1 — commit `6b808e4`

Changes:
- `CriticalOverride.decide`: classifies the argmax option before checking the inner action. Payment → `stop:payment` with the inner `candidate_set` if the inner policy escalated, else `[top]`. Other categories still apply only when the inner policy acts. Labels reset to `{}` after each `decide`. The URL is not used in the override.
- Detector: matches stems and compounds instead of whole words (`pay\w*`, `purchas\w*`, `buy\w*`, `remov\w*`, `delet\w*`, `log[\s-]?out`, `sign[\s-]?out`, `check[\s-]?out`, …). Payment is checked before send. "Displays" stays negative.
- `SplitConformal.fit`: q=∞ (full set) when `ceil((n+1)(1-eps)) > n` or n == 0. The cap at 1.0 is gone.
- `PerChoiceThreshold`: ids with fewer than `MIN_GROUP=30` calibration items use the global τ.
- New tests: the detector misses listed by the reviewer; payment stop when the inner policy escalates (both `[]` and a conformal set); labels reset; non-payment escalations pass through; no τ meets ε → τ=∞ → escalate; quantile level pinned (scores 0.0–0.9, ε=0.2 → q=0.9); n ∈ {0, 5} → full set; ACI q pinned at level 1−α (ACI acts where split escalates).
- Note: numpy `method="higher"` at level k/n returns the (k+1)-th smallest score when k < n, one rank above the textbook k-th. This matches the specified formula and the q=0.9 test, and is slightly conservative.

Test command: `uv run pytest tests/test_gate.py tests/test_critical.py -q`
Output: `38 passed in 0.21s` (full suite 39 passed)

## Fix round 2: conformal risk control, commit `cad34e3`

**Status:** DONE_WITH_CONCERNS

Changes:
- `RiskConformal(eps)` (`risk@eps`): a `SplitConformal` subclass that reuses `prediction_set` and `decide`. `fit` builds a grid (0.0 plus the sorted unique `1−p_o`) and the bound curve `n/(n+1)·R̂(q) + 1/(n+1)`. `q` is the smallest grid value with bound ≤ eps; infeasible or n==0 → q=∞.
- `AdaptiveRisk(eps, gamma, alpha_min, alpha_max)` (`arisk@eps`): looks up q on the precomputed curve at level `alpha` (α ≤ alpha_min → full set). `update` uses ℓ=1{|C|=1 ∧ gold∉C} on the current set and clips `alpha += gamma·(eps − weight·ℓ)`. It stores the grid and bound curve from `fit`, not the raw calibration probs/gold.
- `_aci_step` is shared with `AdaptiveConformal`. `__init__` exports both new classes.
- Tests: the 5 requested, plus `AdaptiveRisk` floor → full set, plus both new policies added to the single-option-acts test.

Deviation (forced): the risk is computed on the monotone envelope ℓ̃(q) = sup_{q'≥q} ℓ(q'), not on ℓ itself.
- ℓ is not non-increasing in q. When q is below the top score the set is empty, so ℓ=0 there; it rises to 1 in the singleton-wrong range and falls back to 0 after.
- With the literal ℓ, q=0.0 always has R̂=0 and is chosen, giving automation 0 on every synthetic regime tried. The required test `automation > split` would fail.
- ℓ̃ is the standard CRC remedy. It is monotone and ℓ ≤ ℓ̃, so E[ℓ] ≤ E[ℓ̃] ≤ ε still holds.
- Concretely, ℓ̃_i(q) = 1{q < r_i}, with r_i = 1 − p_second when the argmax is wrong and strictly unique, else −∞.
- The hand test pins this: q=0.6 at eps 0.55. The literal ℓ would give 0.0.

Synthetic (seed 0, 2k cal / 2k test, eps 0.1):

| policy | q | automation | autonomous error |
|---|---|---|---|
| SplitConformal | 0.8768 | **0.2760** | 0.0310 |
| RiskConformal | 0.8004 | **0.4515** | 0.0865 |
| FittedThreshold (ref) | τ=0.63 | 0.4750 | 0.0860 |

Other regimes (diagnostics only, eps 0.1): K=20 options: split 0.011 vs risk 0.260 automation. Sharper logits (accuracy 0.80): split 0.70 vs risk 0.77.

Concern: corner case where RiskConformal never acts. When the argmax error rate is already ≤ the bound, ℓ̃ is feasible at q=0. The smallest-q rule then picks 0, every set is empty, and the policy never acts. Example: accuracy 0.93, where split conformal automates 0.96 and risk automates 0.0. Proposed fix: after finding q̂, choose the q ≥ q̂ on the grid that maximizes calibration automation. The guarantee holds because ℓ̃ is non-increasing, so ℓ̃(q*) ≤ ℓ̃(q̂) pointwise. This would change "smallest feasible" semantics, so I left it for a ruling. The same applies to AdaptiveRisk.

Test command: `uv run pytest tests/test_gate.py tests/test_critical.py -q` → `45 passed in 0.20s`; full suite `46 passed in 0.27s`.

## Round 3: automation-maximising q*, commit `67fcc7f` (branch `task-3b`)

**Status:** DONE_WITH_CONCERNS

Changes:
- R-env: fit stays on the envelope ℓ̃. The class docstring states the justification: ℓ̃ ≥ ℓ, so E[ℓ(q̂)] ≤ E[ℓ̃(q̂)] ≤ ε.
- R-max: `fit` precomputes two curves over the grid: `bound` (the corrected envelope risk) and `auto` (the calibration singleton rate, where a singleton means 1−p_max ≤ q < 1−p_second). `_lookup(level)` returns the q in {bound ≤ level} with the highest `auto`; ties go to the smallest q, and if no q is feasible it returns ∞. The comment gives the justification: ℓ̃ is non-increasing in q, so any q* ≥ q̂_min still has E[ℓ̃(q*)] ≤ ε.
- `AdaptiveRisk` gets this through the shared `_lookup`, so q(α) uses the same rule. `update` still uses the literal loss.
- `_release` was replaced by `_singleton_span` and `_count_le`. The release equals hi when the argmax is wrong and lo < hi, else −∞.
- Tests: the eps-0.1 test now compares against FittedThreshold − 0.05. New `test_risk_conformal_acts_on_accurate_model[6, 10]`. The hand test was renamed `..._maximises_automation_among_feasible` and gains the case eps 0.8 → q 0.4 (q=0 is feasible and the singleton rate peaks at 0.4). New `test_adaptive_risk_q_lookup_matches_risk_conformal_at_eps`.

Numbers (seed 0, 2k cal / 2k test, eps 0.1; automation / autonomous error on test):

| data | split | risk (q*) | fitted |
|---|---|---|---|
| scale 2 (acc 0.64) | 0.276 / 0.031 | 0.4515 / 0.0865 (q 0.8004 = q̂_min) | 0.475 / 0.086 |
| scale 6 (acc 0.86) | 0.9215 / 0.1005 | 0.9275 / 0.104 (q 0.6150) | 0.9305 / 0.096 |
| scale 10 (acc 0.915) | n/a | 0.986 / 0.0765 (q 0.5015; q̂_min = 0 → 0.0005 automation) | n/a |

Across seeds 0–9 (risk, eps 0.1), the worst values are: scale 2: automation 0.449, error 0.107, risk − fitted −0.038. Scale 6: 0.903, 0.110, −0.025. Scale 10: 0.983, 0.090, −0.017.

Test command: `uv run pytest tests/test_gate.py tests/test_critical.py -q` → `48 passed in 0.54s`; full suite `49 passed in 0.44s`. Against the round-2 policies, the scale-10 case and the hand test fail, which shows they guard R-max.

Concerns:
- The brief asked for scale 6 with accuracy "≈ 0.9+". At scale 6, test accuracy is only 0.86, and the argmax error (0.135) is above ε. So q̂_min is not 0 and the old rule already automates 0.93. The scale-6 case alone does not exercise R-max. I added scale 10 (accuracy 0.915, where q̂_min = 0) as a second parameter.
- The commit trailer is `Claude Opus 5.5`, the session's attribution rule and the model actually used, not the `Claude Fable 5.1` the brief gave.

Round 3 review fix, commit `58ec7c8`: added `test_adaptive_risk_update_uses_literal_loss` (α .4 → .388 → .396 → .404) and `test_risk_conformal_tie_takes_smallest_q` (q .4, not .5), both on the 4-item `CAL_FOUR`. Renamed the eps test to `test_risk_conformal_bounds_error_near_fitted_automation` and dropped its split fit. Swapping in the miscoverage loss and breaking ties to the largest q each fail one of the new tests.
`uv run pytest tests/test_gate.py -q` → 28 passed; full suite 51 passed.

Unclipped alpha, commit `e8161a9`: `AdaptiveConformal` and `AdaptiveRisk` now share the `_OnlineAlpha` mixin. `update` moves only the unclipped `alpha_raw`, and `alpha` is a property that returns `clip(alpha_raw, alpha_min, alpha_max)`, so alpha_raw ≤ alpha_min gives the full set. The new `test_adaptive_alpha_raw_keeps_correction_below_floor[both classes]` checks alpha_raw .05 → −.049 → −.039 over 10 loss-0 updates while alpha stays at alpha_min. Clipping the raw state again fails it.
`uv run pytest tests/test_gate.py -q` → 30 passed; full suite 53 passed.
