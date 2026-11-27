# SDD ledger — plan: docs/superpowers/plans/2026-11-06-surety-m1.md

Spec: docs/superpowers/specs/2026-11-06-surety-design.md (read). Branch: m1 (from main b105aaa).

## Preflight scan

| Tasks | Shared interface | Produces vs consumes | Finding |
|---|---|---|---|
| 0 ↔ 1–8,11 | `surety.types` | Option/DecisionState/ChoiceAnswer/GateDecision/Instance defined in Task 0, consumed everywhere | consistent |
| 2 ↔ 6 | `Backend.choice(state, n_perm, seed) -> ChoiceAnswer` | collect() calls choice with n_perm | consistent |
| 2 ↔ 7 | `TemperatureScaler.fit/apply` | analyze variants "debiased+T" | consistent |
| 3 ↔ 7 | `GatePolicy.fit/decide/update`, `GateDecision.candidate_set` | analyze fits on cal rows (dict probs + gold id) | consistent |
| 3 ↔ 8 | `SplitConformal` candidate_set | h3 uses C as candidate_ids | consistent |
| 4 ↔ 8 | `Supervisor.choose(state, candidate_ids) -> (id|None, Usage)` | h3 script | consistent |
| 5 ↔ 6 | `compile_instance(inst, mode) -> DecisionState` with `meta["gold_id"]` | collect writes gold_id from meta | consistent |
| 6 ↔ 7 | JSONL row keys | analyze reads id/split/gold_id/option_ids/probs/runs | consistent |
| each task | self-consistency | tests reference only names the task defines | clean |

Ruling: Tasks 1–5 touch disjoint files, so they run as parallel implementers in isolated worktrees (one branch each, merged by the controller) — serial dispatch would idle the session while model downloads finish — cost if wrong: a merge conflict to resolve by hand.
Ruling: WebGuard critical-detector evaluation (spec Safety) is not in this plan; deferred to M2 with the live harness — cost if wrong: detector recall unmeasured at M1.
Ruling: Mind2Web secondary dataset deferred to M2 (6 GB download, 13 GB free disk) — cost if wrong: M1 rests on LLM-labeled data only.

## Progress

Task 0: complete (commit b105aaa, controller-implemented scaffold; 1 test passing)
Task 1: minor (deferred): test_autonomous_error_counts_only_singletons also asserts automation_rate; ECE p_max==1.0 clamp and multi-bin sum untested; empty input → NaN with RuntimeWarning; argmax ties resolve by dict order.
Task 1: complete (commits b105aaa..0221189, review clean; merged into m1)
Task 2: implementer DONE, branch task-2 commit 8275a9d, 11 tests; review pending
Task 4: implementer DONE, branch task-4 commit 9cf3025, 8 tests; review pending
Ruling: Clef server launch (scripts/serve_clef.sh) was denied by the auto-mode classifier (external code); M1 runs on Kev-4B only unless the user starts Clef manually — cost if wrong: one executor instead of two in M1 tables.
Task 4: ⚠️ DecisionState positional order — resolved by controller: types.py order is (state_text, instructions, options, meta), tests match.
Task 4: minor (deferred): "NONE, not a1" parses as a1 (NONE not checked first); candidate id absent from state.options still accepted by _parse; `-E1` rejected by lookbehind; no parametrized parse test for E1/E12, punctuation, HTTP error.
Task 4: complete (commits b105aaa..9cf3025, review clean; merged into m1)
Task 2: review — spec ✅, 2 Important (loose temperature assertion; httpx.Client never closed), minors (zero-sum fallback untested, permute input_tokens unasserted, seeded order not pinned, inline import). ⚠️ permute `order` ignored — resolved by controller: Kev serve.py returns runs[i]["probabilities"] keyed by option name; order is informational.
Task 2: fix round 1/5 dispatched to original implementer (2 open)
Task 2: fix round 1/5 (2 addressed per implementer, re-review dispatched; commits 8275a9d..40342f6)
Task 3: implementer DONE_WITH_CONCERNS (keyword recall gaps; url semantics; ACI quantile uncorrected), branch worktree-agent-a542998465fe6ada9 commit e88e753, 24 tests; review dispatched
Task 2: fix round 1/5 (5 addressed, 0 open; commits 8275a9d..40342f6)
Task 2: minor (deferred): permute input_tokens test does not distinguish sum from single value; tests construct clients without closing.
Task 2: complete (commits b105aaa..40342f6, review clean; merged into m1)
Task 7: dispatched (worktree task-7 from c2d0315 + gate branch merged in)
Task 3: review — spec ❌ (payment stop skipped when inner policy escalates), Important: keyword recall gaps; quantile cap at 1.0 breaks guarantee for small n; quantile level unpinned. Minors: labels persist across decide; URL never passed; per-choice groups unbounded; n=0 ZeroDivisionError; no-τ and ACI-q tests missing.
Ruling: critical-action keywords match stems/compounds (pay\w*, log[\s-]?out, …) instead of the brief's literal whole-word list — spec says detection is tuned for recall — cost if wrong: more false escalations on benign labels.
Ruling: conformal quantile is ∞ (full set) when ceil((n+1)(1−ε)) > n or n == 0, replacing the plan's "capped at 1.0" — the cap silently broke coverage for n < (1−ε)/ε — cost if wrong: none (strictly safer).
Ruling: PerChoiceThreshold falls back to the global τ for ids with < 30 calibration items — beyond the brief, prevents τ=0 on one-item groups — cost if wrong: baseline slightly more conservative than JevAgentHarness-style fitting.
Task 3: fix round 1/5 dispatched to original implementer (4 Important + minors open)
Task 5: implementer DONE_WITH_CONCERNS (text-format grounding ids clickable_N; text rows long; compact keeps extra goal lines), branch worktree-agent-a6766dd2454ae5543 commit f13c29d, 10 tests, validated on 64,122 rows; review dispatched
Task 6: dispatched (worktree task-6 from c2d0315; merges Task 5 branch itself)
Task 5: ⚠️ `11:1` sub-option references not matched in compact element keep-list — resolved by controller: all 64,122 rows compile with correct gold_id (implementer run); affects compact-state fidelity only → deferred minor.
Task 5: minor (deferred): banker's rounding can give 0 held-out sites for tiny site counts; under-filled plan splits are silent; AP text fallback labels not truncated; `_element_line` hard-indexes keys; no test for colon-style references or the cap; positional Instance construction in tests.
Task 5: complete (commits b105aaa..f13c29d, review clean; merged into m1)
Task 11: dispatched (own worktree task-11 from m1 + gate branch)
Task 3: fix round 1/5 (implementer reports 4 Important + minors addressed; commits e88e753..6b808e4; re-review dispatched)
Task 7: implementer DONE_WITH_CONCERNS, branch task-7 commit 4c35afe, 57 tests. Concerns: (1) Task 2 TemperatureScaler.fit mis-fits when option ids differ per row → Task 2 reopened, fix round 2 dispatched; (2) ACI audit rule as specified skips unaudited singletons (conservative); (3) eval/__init__.py possible conflict; (4) Opus trailer.
Ruling: ACI online rule applies the ε drift every step — unaudited singleton steps call `update(..., weight=0.0)` (α += γ·ε) instead of skipping — the plan's "else no update" biased the stream toward hard steps and cost automation — cost if wrong: slightly less conservative error control under shift.
Task 7: review dispatched
Task 3: fix round 1/5 (4 Important + 5 minors addressed, 0 open; commits e88e753..6b808e4)
Task 3: minor (deferred): quantile "higher" at k/n returns the (k+1)-th score (slightly conservative); labels consumed per decide (callers must set_labels each step); AdaptiveConformal with empty scores and alpha>alpha_min path unchecked.
Task 3: complete (commits b105aaa..6b808e4, review clean; merged into m1)
Task 6: implementer DONE_WITH_CONCERNS, branch task-6 commit 0cb15ca, 42 tests; smoke 6 live rows: 5/6 correct, latency 2.4–59 s (32–64 options → 47–59 s), input_tokens 0 on permute path; review dispatched
Task 2: fix round 2/5 (per-row-id temperature fit; commit 40342f6..f5ddee1; re-review dispatched)
Ruling (Task 11): confident `none_of_these` acts are trusted (scroll, counts toward done streak); re-ask only on escalate-path none/None; stuck = 3rd identical action escalates, 4th stops; env.terminated → stop — matches the specified tests — cost if wrong: live-loop behaviour tweaks later.
Task 7: review dispatched (opus)
Task 6: review — spec ✅; Critical: torn last line breaks resume; Important: systematic failures not surfaced, --limit samples only cal, resume keyed by id only; minors (duplicate ids, root mismatch message, progress counts, exit 1 on fully-resumed run).
Ruling: collection processes splits round-robin so partial runs contain every split — the single-executor run is ~4 h and analysis must work on partial data — cost if wrong: none.
Ruling: make_plan `--max-options 64` excludes the ~10% of grounding instances with >64 candidates from M1 (each took ~60 s on Kev-4B) — documented in results — cost if wrong: M1 grounding numbers are on ≤64-candidate pages only.
Ruling: instances with >16 options use 3 permutations instead of 5 — latency — cost if wrong: slightly noisier debiasing on grounding rows.
Ruling: fully-resumed run exits 0 (plan said non-zero when nothing written) — shell-friendliness — cost if wrong: none.
Task 6: fix round 1/5 dispatched (1 Critical + 3 Important + rulings)
Task 2: fix round 2/5 (1 addressed, 0 open; commits 40342f6..f5ddee1)
Task 2: complete again (merged f5ddee1 into m1); minor (deferred): gold absent from a row's ids raises bare ValueError; empty probs → NaN.
Ruling: supervisor for M1 = `gpt-oss:20b` served by the user's existing Ollama (OpenAI-compatible at :11434/v1) instead of downloading Qwen3.6-35B-A3B (20 GB; 19 GB free) — zero download, reasoning model, already resident — cost if wrong: supervisor weaker than the spec's default; swap later via `--base-url/--model`.
Task 6: fix round 1/5 implementer DONE (4 findings + R1–R3 done; commit 0cb15ca..2f35372); R4 one-liner requested before re-review
Task 7: review — spec ✅; Important: 3 weak tests (ACI/H1/H2 assert order or ≤eps only, pin numbers); cross-task ⚠️: SplitConformal/ACI calibrate miscoverage, not the autonomous-error target → automation 0.11 vs fitted 0.33 at eps 0.05 on synthetic.
Ruling (design, spec §Escalation): add conformal risk control (CRC, Angelopoulos et al. 2022) on the autonomous-error loss ℓ(q)=1{|C_q|=1 ∧ gold∉C_q}: q̂ = smallest q with (n/(n+1))·R̂_n(q) + 1/(n+1) ≤ ε → `RiskConformal(eps)`; online `AdaptiveRisk(eps, gamma, …)` with ℓ_t known (0) on non-singletons, observed (weight 1/r) on audited singletons, weight 0 otherwise. `SplitConformal` (coverage, LAC) stays as the baseline and supplies the candidate set shown to the supervisor (H3). Act rule = RiskConformal singleton; escalate view = LAC set. Cost if wrong: guarantee holds in expectation over calibration draws, not per-set; one more policy row in every table.
Task 3: reopened for the CRC policies (fix round 2 dispatched)
Task 6: R4 one-liner done (commit 5919b62); re-review dispatched over 0cb15ca..5919b62
Task 6: fix round 1/5 (6 addressed, 0 open; commits 0cb15ca..5919b62)
Task 6: minor (deferred): no tests for backend-mismatch warning, CLI interleaving, --root mismatch; itertools import mid-block; --max-options 0 treated as unset.
Task 6: complete (commits 493f627..5919b62, review clean; merged into m1). M1 collection running from the task-6 worktree (same code) → runs/kev_compact.jsonl
Task 11: implementer DONE_WITH_CONCERNS (no browser run; conformal policy unfitted in CLI; BrowserGym unpinned), branch task-11 commit 11308f8, 94 passed 1 skipped; review dispatched
Ruling: fine-tune trigger treated as fired on early collection numbers (Kev-4B zero-shot 30–67% correct, mean p_max 0.23–0.40 on the first 59 rows) → Task 12 added: Kev LoRA training data from calibration-site instances disjoint from all plan rows and held-out sites, plus a launch script (training itself runs after the collection finishes, one GPU job at a time) — cost if wrong: a day of prep if the full numbers turn out fine.
Task 12: dispatched (own worktree task-12 from m1 e6fb892)
Task 3: round 2 implementer DONE_WITH_CONCERNS (commit cad34e3): CRC fitted on the monotone envelope of the autonomous-error loss (literal loss is a bump 0→1→0 in q, not monotone); synthetic eps 0.1: split automation 0.276 / risk 0.452 / fitted 0.475.
Ruling: RiskConformal fits on the envelope ℓ̃(q)=max_{q'≥q}ℓ(q') (envelope ≥ loss ⇒ bound holds) and then picks the automation-maximising q* ≥ q̂_min (pointwise monotone ⇒ E[ℓ̃(q*)] ≤ ε) — the literal smallest-feasible rule never acts on accurate models — cost if wrong: the guarantee is in expectation over calibration draws, slight optimism from maximising on the calibration set.
Task 3: round 3 dispatched (R-max for RiskConformal and AdaptiveRisk)
Task 3: round 3 re-dispatched to a fresh implementer (original agent unresumable: worktree unverifiable) in worktree task-3b from cad34e3
Task 11: review — spec ✅; Important: no payment check on supervisor-chosen actions; duplicate full-view re-ask on empty candidate_set; combobox→select mapping; no viewport filtering (scroll never changes options); fill+Enter on every type. Minors: None slot → fill(""); slot_value uncounted; query string in traced url; <a> role; checkbox value; no timeouts; BrowserGym roles/truncated; href unused; port 8080.
Task 11: fix round 1/5 dispatched (5 Important + minors)
M1 collection complete: 1,600 rows (cal 600 / test_in 400 / test_shift 600), Kev-4B compact, 2026-11-06 ~15:40. Prelim analysis (results/m1/kev_compact_prelim.md): acc 0.48/0.46; ECE first_run 0.17/0.11 → debiased 0.17/0.13 (no gain) → +T 0.086/0.077 (halved); fitted@0.05 auto_err 0.040 in-dist → 0.060 on shift (drift); per_choice 0.03/0.048; coverage-conformal auto_err 0 but automation 0.03–0.05; ACI(coverage) identical to static. Automation ≤ 0.34 overall → fine-tune trigger confirmed.
Task 3: round 3 DONE (fresh implementer; commit 67fcc7f on task-3b; synthetic eps 0.1: risk 0.452 vs fitted 0.475 automation; accurate-model case automation 0.986); review of rounds 2+3 dispatched over 6b808e4..67fcc7f
Task 7: fix round 1 dispatched (pin tests; w=0 ruling; add risk/arisk rows) on task-7 worktree with task-3b merged
Task 11: fix round 1/5 implementer DONE (commit 11308f8..07beda3, 111 passed); re-review dispatched
Task 12: implementer DONE (commit 062f4aa; 8000/500 records; Mac training time unknown — try --max_steps 5 first); review dispatched
Task 11: fix round 1/5 (6 addressed, 0 open; commits 11308f8..07beda3)
Task 11: minor (deferred): non-editable role=combobox/listbox divs map to select and fail with `error`; Playwright SELECTOR misses [role=searchbox]/[role=combobox] divs; scroll/back have no timeout and observe() exceptions unhandled; BrowserGym `checked` stringification; MAX_ELEMENTS 150 vs compiler cap 40; BrowserGym commit not pinned; no browser run yet.
Task 11: complete (commits 82a073a..07beda3, review clean; merged into m1)
Task 12: review — spec ✅; Important: heldout drawn from training sites (not independent); minors (curl timeout, readiness loop, trap/kill, port override, private _draw, shortfall, main guard).
Ruling: Kev fine-tune heldout = whole sites reserved from the training-site pool — same-site heldout only measures memorisation — cost if wrong: slightly fewer training sites.
Task 12: fix round 1/5 dispatched
Task 3: CRC review (rounds 2+3) — spec ✅, math verified by hand (envelope, bound, lookup monotone, no leakage); Important: literal loss untested in AdaptiveRisk.update; minors: tie-break untested, test name. Fix round dispatched.
Task 3: CRC fix round (tests) DONE commit 58ec7c8; re-review dispatched
Task 12: fix round 1/5 implementer DONE (commit b85672e; heldout = 5 whole sites, two dominate). Ruling: per-site heldout cap 60 → ≥ 9 sites — cost if wrong: none. Follow-up requested before re-review.
Task 7: fix round 1/5 implementer DONE (commit d6d1a19; 104 tests). Real data (results/m1/kev_compact.md): risk@0.05 auto_err 0.028/0.033 at automation 0.198/0.222 (in/shift); risk@0.1 0.058/0.047 at 0.268/0.283; arisk r=0 drifts to alpha 0.5 (err 0.20–0.28); r=0.05 over target; r=0.2 0.058 (eps .05) / 0.095 (eps .1); aci rows zero error, identical across audit rates.
Ruling: adaptive gates keep an unclipped alpha_raw and clip only for the threshold lookup — clipping lost 1/r-weighted corrections (synthetic r=0.2 error 0.181 vs 0.094) — cost if wrong: none (standard ACI). Sent to Task 3 (task-3b); Task 7 re-pins afterwards.
Task 8: dispatched (H3 supervisor-in-the-set study on gpt-oss:20b via Ollama)
Task 3: CRC test fixes verified (58ec7c8, all addressed); unclipped-alpha change pending on task-3b before merge
Task 12: fix rounds done (b85672e, 2fd2f84; heldout 27 sites ≤60 each); re-review dispatched
Task 12: fix rounds 1–2/5 (all addressed; commits 062f4aa..2fd2f84)
Task 12: minor (deferred): test_in shares sites with training (in-dist numbers inflated by design — report both); EXIT trap does not cover kill -9.
Task 12: complete (commits e6fb892..2fd2f84, review clean; merged into m1). Training not yet run (GPU busy with the full-context ablation; try `scripts/train_kev.sh --max_steps 5` first).
Task 3: unclipped alpha DONE (commit e8161a9); re-review dispatched. task-3b merged into task-7 worktree for Task 7 fix round 2.
Task 7: fix round 2 dispatched (re-pin after alpha change, rerun real data, add compact-vs-full ablation table)
Task 3: alpha fix re-review clean (e8161a9). Task 3 (CRC extension) complete: commits 6b808e4..e8161a9 merged into m1.
Task 8: code DONE (commit 77798c9, 106 tests); real run blocked: gpt-oss reasoning exhausts max_tokens (empty content), 50–65 s/call with the GPU shared.
Ruling: supervisor sends "Reasoning: low" system line for gpt-oss, max_tokens 1024, reasoning-text fallback parse flagged in Usage; H3 n=100 — cost if wrong: a weaker supervisor than medium/high reasoning; reported in the table.
Task 7: fix round 2/5 implementer DONE (commit cd3f3e4; 109 tests). Real data arisk r=0.2 on test_shift: eps .05 → 0.047 (tail 0.045, automation 0.247); eps .1 → 0.082 (tail 0.035, automation 0.320). Ablation (35 rows, unmatched n_perm): compact acc 0.371 / full 0.314; full 3.6× slower. Scoped re-review dispatched over both rounds.
Task 7: re-review of rounds 1–2: 5 addressed, 1 open (load_collection drops unterminated complete last line) → round 3 dispatched
Task 7: round 3 DONE (b17ad8c); scoped re-review dispatched
Task 7: round 3 re-review clean (b17ad8c). Task 7: complete (commits 120bb5a..b17ad8c, review clean; merged into m1).
Task 7: minor (deferred): ablation compares mismatched n_perm (compact mean 4.3 vs full 3) on a small n; a torn last line that is itself valid JSON would be kept.
Task 8: relaunched with --reasoning low --max-tokens 1024 --n 100 (commit 2abc570; one-row check: full→DONE correct 20.7 s; set→NONE abstain). Ruling: keep the NONE allowance; h3 table reports abstain rate and accuracy given an answer — cost if wrong: none.
Task 8: abstention columns DONE (3019849); review dispatched over d6d1a19aff5c50d13aa4cf3b9c8c3ff7d4795a40..3019849
Task 8: review — spec ✅; Important: fallback parse takes the first id; abstain conflates parse failure; h3 table test weak. Fix round 1 dispatched (job relaunch with resume after the fix).
Task 8: fix round 1 DONE (81ebb61) + merge of m1 resolved (180337a; 156 passed). H3 real data: full acc 0.268 (unparsed 0.464, abstain 0.041, acc_given_answer 0.542, 1988 prompt tok, 19.9 s); set acc 0.223 (unparsed 0.563, abstain 0.097, acc_given_answer 0.657, 1495 tok, 15.0 s); gold_in_set 0.816. Ablation (60 rows): compact acc 0.450 / ECE 0.177 / 5.8 s vs full 0.417 / 0.189 / 24.0 s. Scoped re-review dispatched.
Task 8: fix round 1/5 (all addressed incl. merge; commits 3019849..180337a)
Task 8: minor (deferred): finish_reason not recorded per row (truncation skip unlogged); different n per condition (97 vs 103) from skipped/errored calls; pre-fix rows lack the `abstained` flag.
Task 8: complete (commits d6d1a19..180337a, review clean; merged into m1)
Task 10 (docs: README + docs/results/m1.md): dispatched (worktree task-10)
Final whole-branch review dispatched over b105aaa..180337a (opus)
Task 10: complete (commit 2a8e0d8; merged into m1). minor (deferred): absolute local path in docs/results/m1.md collection line; H3 n unpaired (6 Ollama 500s + 3 resumed rows)
Final review (b105aaa..180337a): needs fixes — Critical: runner's payment check only on supervisor picks (unwrapped policies act on payment); Important: alpha_raw unbounded above; reasoning-text fallback leaks into slot_value; RiskConformal escalates with CRC set not LAC; h3_table unpaired; no credentials stop; Minor: quantile off-by-one (conservative), T and gates fit on same cal rows, no CI, userinfo in traced url, .claude/ not ignored, gold_in_set label, set listed by probability (position confound).
Ruling: Kev fine-tune on Apple MPS abandoned (5-step trial logged no step in 20 min; transformers reference kernels) — use Kev's Modal H100 recipe (~$1/run) in M2 — cost if wrong: none.
Ruling: alpha_raw clipped on the upper side only (keeps the ACI bound); RiskConformal.decide escalates with the LAC set at the same ε; H3 paired on ids present in both conditions and re-run into a fresh file (resumed rows used the rejected first-id fallback); credentials category + role `password` → `stop:credentials`; T and gates stay fitted on the same cal rows for M1 but documented as in-sample — cost if wrong: finite-sample statement slightly optimistic.
Final fix wave dispatched (worktree final-fix)

## Overnight 2026-11-16 (Windows 11 laptop: i7-11800H, 16 GB RAM, RTX 3050 Ti 4 GB; branch overnight-2026-11-16)

Setup: uv 0.5.24, 183 passed 1 skipped as on the Mac; WebJev 118 MB; `make_plan.py --seed 0` byte-identical to artifacts/m1/plan.json; `make_kev_train.py` stats identical to artifacts/m1/kev_ft_stats.json (paths aside).
Final fix wave (2a8e0d8..85aa7e5) re-reviewed by an independent agent: critical-check, alpha cap, slot_value, LAC escalation, paired H3, k-th quantile, Wilson CI, userinfo, labels all OK; credentials stop INCOMPLETE. Defects D1–D10 found; all fixed (eb996cb, 4872ab4, 7423975).
Ruling: RiskConformal escalates with the full list when its LAC set has < 2 options (a one-element LAC set is the executor's own pick; H3 only tested |C| ≥ 2) — cost if wrong: more supervisor tokens on those steps.
Ruling: select actions classify label + chosen value ("Bulk actions" → "Delete selected") — cost if wrong: extra critical flags on benign dropdowns.
Ruling: WebGuard target text from boyuanzheng010/webguard `annotation_folder` (row-aligned with osunlp/WebGuard after dropping BUG rows: 5,999/5,999 identical on url, tag, site, label); label = reviewed label; first-pass label discarded — cost if wrong: recall measured on text cut to 30 chars.
Ruling: WebGuard-motivated keyword families (submit, sign up/register, save, download/install, Spanish) tabulated in results/webguard/detector.md, not adopted — "submit" would escalate every MiniWoB submit; owner decides (or builds the spec's Kev noul detector) — cost if wrong: detector HIGH recall stays 0.224.
Ruling: uncertainty without re-collection: cal re-drawn from cal ∪ test_in (row level, like the plan's split) 200×, T and gates refit; site-cluster bootstrap of the original fit; 200 adaptive streams — cost if wrong: spread reflects the fixed 1,600 rows only.
Ruling: H3 read as inconclusive, not refuted: on the 26 ids both conditions answered, full 0.654 vs set 0.500, exact McNemar p=0.219; the headline gap is mostly parse failures (0.579 vs 0.368 unparsed) — cost if wrong: none (both numbers are reported).
Ruling: supervisor JSON-constrained answers (`--json`, response_format json_schema, id ∈ allowed ∪ {NONE}) and `--api-key-env` added as opt-in; M1's H3 setup stays the default — cost if wrong: none.
Ruling: local Kev serving abandoned on this laptop: Kev-0.8B (CUDA torch 2.8, reference Qwen3.5 kernels, no Triton on Windows) grew the pagefile ~4 GB within seconds of its first request and C: fell to 0.4–0.7 GB free twice; the Kev environment and model cache made tonight were deleted (C: back to ~9 GB free; ~3 GB of pagefile growth returns at reboot) — cost if wrong: no second executor and no local fine-tune tonight.
Ruling: live runner = offline gate fitted on a collection's cal rows (loop.fit, `--fit`), T applied before the gate, online update every step as offline (escalation label w=1; random audit of acted steps at rate r, w=1/r, audit answer acted on; else w=0) — cost if wrong: audits cost r × supervisor calls on acted steps.
Ruling: BrowserGym input types and fallback names (placeholder, name, id) read from the DOMSnapshot by bid: Chrome's AX tree reports MiniWoB's password input as an unnamed textbox — checked live on login-user (stop:credentials fires; click-test reward 1.0 with scripted models) — cost if wrong: none.
Ruling: Kev-4B fine-tune launcher on Modal (scripts/modal_train_kev.py) reuses Kev's modal_app image and passes --max_state 7552 (Kev's kev-finetune skill keeps the 384-token default, which drops ~94% of WebJev records); init pinned to hub main 6cfce5c2 (unchanged since 2026-10-01, what M1 served) — cost if wrong: untested until run on the owner's account.
Ruling: BrowserGym observation keeps clickable, visible nodes outside ARIA roles (role `clickable`, or their own role such as image/LabelText), named by their own text (not descending into other options) or DOM attributes (aria-label, alt, title, placeholder, name, id, class); unnamed wrappers dropped. MiniWoB++ sweep (125 tasks, no models): zero-option tasks 27 → 4 — cost if wrong: more options per step (max 30 → 30; click-checkboxes-large 11 → 20) and class names as labels for icons.
Ruling: H4 task list = 68 MiniWoB++ tasks with ≥ 3 options that need no dragging, drawing, coordinates or colour perception (configs/miniwob_h4.txt); login-user, login-user-popup, enter-password, buy-ticket are safety checks (configs/miniwob_safety.txt), excluded from H4 success — cost if wrong: H4 on an easier, text-centric subset; state it in the report.
Ruling: H4 runner keeps one gate per system for its whole stream (seed-major, then task), so arisk's α carries across episodes; audits only for arisk — cost if wrong: none.
Second review (independent agent, 006f180..b80100d): core paths verified (one gate update per step on every path, calibrate placement, adaptive_draws = aci_simulation at seed 0, cluster bootstrap, McNemar); defects fixed in 7725606.
Ruling: an explicit supervisor NONE is the label none_of_these (acted on: scroll; counts toward the done streak), not a dropped answer; only unparsable replies give weight 0 — cost if wrong: abstentions end episodes sooner.
Ruling: critical (non-payment) escalations show the full option list, superseding the earlier one-option verify prompt: its answer is the adaptive gate's label and a confirm prompt biases it toward agreement — cost if wrong: more supervisor tokens on critical steps.
Ruling: audit draws use one Random per (system, seed) stream owned by the caller; live scoring uses the fit's orderings (5, 3 above 16 options) whenever --fit is given — cost if wrong: 3–5× executor calls per live step.
Ruling: credential stems need context (PIN not "PIN code"; expiry/expiration + date|month|year|mm; MM/YY); autocomplete cc-*/one-time-code/*-password marks secret inputs; per_choice is not offered live (thresholds keyed by WebJev ids) — cost if wrong: an unlabelled "Expires" field typed without a stop.
Ruling: per-task-type (Mondrian) risk control reported as an analysis, not adopted as the gate: it controls each type on calibration sites (action prediction 0.051 / 0.098 at ε=0.05 / 0.1) but on held-out sites action prediction reaches 0.074 / 0.118 and the combined per-type gate exceeds ε=0.05 in 90% of draws, while the marginal gate never does — the marginal margin came from pooling — cost if wrong: per-type guarantees left to the adaptive gate (M2).
Task 11 deferred minors addressed (156475d): Playwright SELECTOR now covers ARIA widget roles, [onclick], <summary>; non-native combobox/listbox divs click instead of select_option; go_back timeout; observe retried once after a navigation; autocomplete secrets; title/alt/id names. Still open: BrowserGym commit not pinned (uv.lock pins the installed version), checkbox `checked` stringification, MAX_ELEMENTS 150 vs compiler cap 40.
Ruling: per-type control under shift = adaptive risk with one gate per task type (audits 0.2): on held-out streams each type returns to ≈ ε on average (ε=0.05: action prediction 0.054, grounding 0.045, all 0.049), over ε in 30–50% of streams as ACI controls the average — recommended for M2 over the static per-type gate — cost if wrong: per-group gates need enough audited steps per group.
Third review (independent agent, b80100d..b84704e): core paths re-verified (one gate update per step on every path, live auto_err estimator, groups and sweep code); defects fixed in 2399c80, 915aa98 and the env commit after it.
Correction: "H4 runner keeps one gate per system … cost if wrong: none" was false on resume (α reset to ε, audit stream restarted); resume now restores α and the audit/shadow random streams from the file, refuses other settings, and retries errored episodes.
Ruling: a clickable <label> wrapping or pointing at a kept control is not an option; its text names the control when the control has none — a duplicate made an audit naming the label count as an error (weight 1/r) — cost if wrong: none.
Ruling: one option per browsergym bid (the first usable node); ROLES nodes keep their old fallback names (placeholder, name, id); the wider list (aria-label, alt, title, class) only names clickable nodes; names capped at 80 chars — cost if wrong: none.
Ruling: H4 table compares each system with supervisor-only on matched (seed, task) pairs and reports the unanswered share of supervisor calls; live scripts default the supervisor to 1,024 max tokens — cost if wrong: none.
Ruling: credential stems: card/ATM/n-digit PIN codes, bare expiry/expiration, MM/YY(YY) not after DD/; snake_case read as words; a NONE inside recovered reasoning text is not an abstention — cost if wrong: WebGuard label+url recall 0.228 (label 0.160 unchanged).
Ruling: H4 default --max-steps 20 (was 10): a scripted oracle needs 7–13 clicks on click-checkboxes-large — cost if wrong: failing systems spend up to 20 executor calls per episode.
Live re-check after the env fix (scripted models, 3 seeds each): click-link, click-button, click-widget, enter-text, click-option, click-checkboxes, click-checkboxes-large all solved (the last with a 16-step budget); MiniWoB sweep: 0 errors, LabelText options 29 → 0, H4 tasks all ≥ 3 options.
Fourth review (independent agent, 2399c80..62491a7): resume state round-trip, one gate update per step, shadow/audit separation and option counting verified; defects fixed in 39d9322, 4eaaa74, 5677171.
Ruling: "error:<Type>" marks an episode that could not run (retried on resume, kept out of the table); the runner's own "error" (the page rejected an action) is an ordinary failed episode — it had been retried forever and dropped, inflating systems — cost if wrong: none.
Ruling: an H4 resume also compares backend, base URL, a hash of the --fit file and every supervisor setting — cost if wrong: none.
Ruling: date/time inputs (Chrome Date/DateTime/InputTime, DOM date/time/month/week) are text inputs named with the ISO format fill() accepts (scripted oracle: enter-date and enter-time 3/3); labels are resolved before options are emitted and fold only into form controls — cost if wrong: none.
Ruling: credential stems read camelCase/snake_case DOM names as words; bare MM/YYYY is a date format, not a card field; left as recall-first trade-offs: an "Expiration" dropdown still stops, "Enter PIN code" (often a postcode) does not — cost if wrong: a stopped share-link form; a typed PIN on a PIN-code-labelled card field without autocomplete.

## 2026-11-24 (restore, research, learned-score gates; branch overnight-2026-11-16, not pushed)

Restore: the repo folder was emptied around 08:14 IST on 2026-11-16 (cause unknown; nothing was pushed). Source, tests and scripts were rebuilt from the session's own edits, docs by replaying every recorded edit onto 006f180 (README byte-identical to the 2026-11-16 one), results regenerated from artifacts/m1 and checked against the recorded numbers, MiniWoB++ coverage re-run in Chromium (identical counts). 293 passed / 2 skipped, as before.
Ruling: no push (owner, 2026-11-24); a verified git bundle of main + this branch is kept outside the repo (C:\Users\DELL\Documents\surety-backups) and refreshed after each work block — cost if wrong: a second folder loss would cost the work since the last bundle.
Ruling: LearnedRisk uses five features of the probability vector only (p_top, margin, normalised entropy, log option count, top is terminal), so it runs live without new inputs; adding permutation agreement, the minimum per-ordering top probability and task type gave the same automation on 40 draws (0.297–0.303 at ε=0.05) — cost if wrong: a few points of AUROC left on the table.
Ruling: the error model is cross-fitted (5 folds, out-of-fold scores calibrate τ, test steps scored by the mean fold model) rather than split 50/50: same automation, realized error closer to ε; validity approximate (fold models see 4/5 of cal, and T is fitted on all of cal) — cost if wrong: realized error drifts above ε by the train/score mismatch; measured over 200 draws in results/m1/kev_compact_learned.md.
Ruling: the H4 opt-in "learned" system is learned_pac (Learn-then-Test, δ = 0.1), not learned@ε (CRC): static risk@ε never exceeded ε on held-out sites over 200 draws, and CRC on a sharper score sits on ε (exceeding it in a large share of draws) — cost if wrong: a few points less automation than learned@ε.
Ruling: alearned defaults γ = 0.005, κ = 0.2 (target 0.8·ε), doubly-robust feedback, picked on M1's test_shift streams from 24 simulated configurations (in-sample) — cost if wrong: tuned to these 600 rows; confirm on new data (H4 or a re-collection) before relying on them.
Ruling: the default H4 systems stay the six of the spec; learned and alearned are opt-in (`--systems ...,learned,alearned`), alearned audited like arisk — cost if wrong: none.
Ruling: the critical-action keyword lists are unchanged (still the owner's decision); research on detectors and confirmation policies is in docs/research/2026-11-24-improvements.md — cost if wrong: HIGH-risk recall stays 0.16 (labels).
Ruling: alearned's noise-sized margin and audited-error barrier (research §3) are API/offline options, not live specs: both need the stream's horizon in advance — cost if wrong: H4 has no per-stream guarantee mode yet.
Review (independent agent, 2026-11-16 fix round + learned gates): 8 defects, 7 fixed in 2706e38 (live γ default, unanswered-audit bias, camelCase credential stems, month/week hints, failed-episode replay, dead browser, label for=); CRC/LTT/DR/resume verified.
Ruling: answered audits weigh 1/(r·q̂), q̂ = the stream's answered/asked audits (missing-at-random assumption), counts saved per system for resume — cost if wrong: if hard steps go unanswered more often, arisk still under-counts errors; --supervisor-json keeps q near 1.
Ruling: the learned model's terminal feature keeps none_of_these as terminal live although WebJev has none (it errs toward escalation) — cost if wrong: more supervisor calls for learned systems in H4 than offline predicts.
Pushed: branch overnight-2026-11-16 to origin on the owner's instruction (2026-11-24, "Commit all this on GitHub"); main untouched, no pull request opened.

## 2026-11-25 (option-prior correction)

Local main and overnight-2026-11-16 reset to the rewritten origin history (same trees; the early commits' author email was changed on GitHub overnight) — cost if wrong: none, the old hashes remain in the reflog.
Ruling: PriorCorrector fits one offset per recurring action id, page-element ids (E12, clickable_12, bid:12, e12) excluded, L2 1.0 fixed before looking at results, damped Newton — cost if wrong: a slightly weaker correction than an unpenalised fit (held-out action-prediction accuracy 0.570 vs 0.59 unpenalised).
Ruling: the correction is not a default anywhere (offline analysis only; live MiniWoB ids are page elements, so it would be a no-op there): on held-out sites the static gates overshoot ε with it and adaptive gates keep little of its gain — cost if wrong: ~4–9 points of automation on calibration-like sites left unused.

## 2026-11-25 (fine-tune prep, while the owner sets up Modal)

Ruling: the fine-tuned Kev trains on and is scored with a new plan, M2 (`artifacts/m2/plan.json`): disjoint training / calibration / test websites (610 / 304 / 304, seed 0), every split spread over its websites (`draw_spread`, at most 4 steps a website). Why: 549 of M1's 600 cal steps sit on sites M1's training file draws from, and M1's top 10 sites hold 36–41 % of each split — cost if wrong: fine-tuned rows are not comparable row for row with M1's; the zero-shot re-score on M2 is the baseline instead.
Ruling: a website is a registrable domain without its public suffix (publicsuffixlist, private suffixes on): en./de.wikipedia.org, apps.apple.com, amazon.de and amazon.com group together; it errs toward merging (weather.com with weather.gov), which only makes the split stricter — cost if wrong: a few unrelated sites share a group.
Ruling: the training file is rebuilt from M2's training websites (8,000 / 500 steps from 529 / 31 websites; 771 / 27 hosts on 2026-11-16), recipe unchanged, and the launcher refuses data from another plan — cost if wrong: fewer training websites than the 2026-11-16 data.
Ruling: scoring runs kev.serve and offline_collect in one Modal container on the training volume (`scripts/modal_score_kev.py`), after a 100-step parity check against M1's rows; a score directory is tied to one checkpoint and plan — cost if wrong: one short wasted GPU run.
Review (independent agent, first version of the prep): 7 findings, all fixed: host-level split leaked 221 of 1,600 cal/test steps onto training websites; misplaced `modal volume get`; resume across checkpoints; cal/test_in grouped by type; no plan check on the training file; no `--detach`; fine-tuned head at T = 1.0 (documented; the gates refit T).
Ruling: M1 stays as published, with a caveat: it holds out hosts, and 89 of its 600 test_shift steps share a website with calibration hosts (report §6) — cost if wrong: M1's held-out-site numbers are somewhat optimistic.

## 2026-11-25 (risk-targeted audits)

Ruling: targeted audits use the Neyman allocation q ∝ √v(ê), v = ê(1 − ê) for doubly-robust feedback or ê for importance weighting, scaled to the budget on the calibration steps the gate acts on and clipped to [floor, 1]; floor and margin are chosen on calibration-site streams only (`scripts/offline_audits.py`) — cost if wrong: none for published numbers (offline option).
Ruling: not wired into the live runner: at the same budget it gains 1.4 points of automation at ε = 0.05 (κ 0.15: 0.281 vs 0.267, 6.5 % vs 9 % of held-out streams over ε) and nothing at ε = 0.1, because acted steps are uniformly low-risk (best-case variance cut 10 %) — cost if wrong: about a point of automation at ε = 0.05 left unused in H4; re-measure after the fine-tune.

## 2026-11-27 (supervisor on free API tiers)

Ruling: H3's stronger supervisor is gpt-oss-120b on Groq's free plan: no card, same family and prompts as the 20b run, strict JSON by constrained decoding; Cerebras's trial needs a card and OpenRouter's free list has no gpt-oss — cost if wrong: H3 rests on an open-weights supervisor, not the spec's frontier model.
Ruling: a call waits out 429s while its waits total at most 900 s, else RateLimited stops the run (exit 75) and the same command resumes; server errors and dropped connections retry 3 times; "request too large" and other 4xx fail the call at once — cost if wrong: an unattended run idles up to 15 min per call near the daily cap.
Ruling: gpt-oss's reasoning level goes to Groq as a request field (`--param reasoning_effort=low`) and the prompt keeps "Reasoning: low"; `params` joins the resume check as None when unset, so earlier files still resume — cost if wrong: none for published numbers.
Ruling: H3 on Groq keeps the 20b run's max_tokens 1024; its longest prompt (~8.1K tokens) cannot fit the free 8K-a-minute cap and two more come within 1,024 of it, so 1–3 full-list grounding calls are expected to fail as too large and leave the paired analysis — cost if wrong: H3's paired sample loses its longest prompts (n 97–99 of 100).
Ruling: `--api-key-env NAME` falls back to a `NAME=value` line in the gitignored `.env`, read only when the variable is unset — cost if wrong: a free-tier key in plain text on the owner's disk.
