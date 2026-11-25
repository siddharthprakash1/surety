# 2026-11-24: restore, research, learned-score gates

Session: Claude Code on the Windows laptop, overnight 2026-11-24 (IST), while the owner slept. Branch
`overnight-2026-11-16`, pushed to GitHub on the owner's go-ahead the next morning; not merged into `main`.

## TL;DR

- **The lost branch is rebuilt.** The repo folder was emptied around 08:14 IST on 2026-11-16, after the overnight
  report was written (cause unknown; nothing had been pushed, so 41 local commits were gone). Everything is back,
  checked against the record: 293 tests pass as before, every results table regenerates to the recorded numbers, the
  rebuilt README is byte-identical to the one written that night, and the MiniWoB++ browser checks reproduce.
- **New gates: more automation at the same error.** A small learned model of "is the executor's top pick wrong?" (five features of the probability vector, cross-fitted) ranks errors on held-out sites at AUROC 0.84 vs 0.70 for the top probability. With a Learn-then-Test threshold it matches risk control's held-out error at ε = 0.05 (0.038 vs 0.037) while automating 27 % of steps instead of 23 % (38 % vs 29 % at ε = 0.1), and it fixes the per-task-type imbalance. Its adaptive version exceeds ε in 9 % of held-out streams instead of arisk's 40 %, at more automation. Ready live as `learned_pac:<eps>` / `alearned:<eps>` and as opt-in H4 systems.
- **Research:** three literature scans (online risk control, selective deferral and scores, web-agent safety), 71 citations, each opened by the agents (one venue unverified), in `docs/research/2026-11-24-improvements.md` with a ranked roadmap. Top items: fine-tune the executor (it caps everything), run H4 with the new gates, a two-tier critical detector (keywords, then an LLM judge on flagged steps), and audits targeted by predicted risk.

## What needs you

1. **Push: done** (2026-11-24, `origin/overnight-2026-11-16`). Merging it into `main` (a pull request) is the next
   call. The local bundle backup of `main` + this branch stays at
   `C:\Users\DELL\Documents\surety-backups\overnight-2026-11-16.bundle`.
2. Unchanged from 2026-11-16: a Modal account for the Kev-4B fine-tune, a hosted supervisor key for H3, the
   keyword decision for the critical detector (now with research behind it, §Research).
3. New, optional: whether ε should bound the joint rate P(act ∧ wrong) (today; ~1 in 4–5 autonomous actions may be wrong at ε = 0.05) or the error among acted steps (selective risk; research §2); and whether to adopt the two-tier detector (research §4).

## Restore: how and what was checked

- Source, tests, scripts and configs: rebuilt from the session's own edits (the transcript holds every write and
  edit), at the final state of 2026-11-16 07:05. `uv run pytest -q`: 293 passed, 2 skipped, as recorded.
- Docs: every recorded edit and shell write replayed in order onto `006f180` (74 operations, none failed); the
  replayed README is byte-identical to the one this session wrote independently from the record.
- Results: `offline_analyze`, `offline_ci`, `offline_sweep`, `offline_groups` and `eval_webguard` regenerated from
  `artifacts/m1/`; every number checked against the record (CI, ε sweep, per-type, WebGuard, H3 line).
- Live: all 125 MiniWoB++ tasks observed again in Chromium (no errors; tasks with no option 4, with one 19, as
  recorded); scripted oracles solve enter-text, click-option, click-checkboxes(-large) 3/3 and
  click-link/button/widget 9/9; `SURETY_LIVE=1` browser tests 3 passed.
- A profile-wide search found no other copy of the lost files. The commit hashes in the 2026-11-16 report refer to
  the lost history; the rebuilt branch carries the same content in fewer commits.

## New: gates on a learned error score

Code: `src/surety/gate/learned.py` (`LearnedRisk`, `learned_pac`, `AdaptiveLearnedRisk`), `src/surety/eval/learned.py`,
`scripts/offline_learned.py`; results: `results/m1/kev_compact_learned.md`, Figure `docs/results/kev_compact_learned_sweep.png`;
write-ups in README ("What held"), `docs/results/m1.md` and the report draft (§3, §5).

Held-out sites, 200 calibration draws, mean auto_err (share of draws over ε) / automation:

| ε | risk (M1) | learned_pac | learned |
|---|---|---|---|
| 0.05 | 0.037 (0.000) / 0.226 | 0.038 (0.005) / 0.274 | 0.046 (0.130) / 0.302 |
| 0.1 | 0.054 (0.000) / 0.294 | 0.078 (0.020) / 0.379 | 0.095 (0.260) / 0.411 |

- Per task type at ε = 0.05, learned_pac: grounding 0.039 (risk 0.059, over ε in 70 % of draws), action prediction
  0.038 at twice the automation.
- A clean protocol (T and model on 400 separate rows, threshold on the 600 cal rows: exact validity) gives the same
  numbers, so the cross-fitting approximation does not matter here.
- Adaptive (200 held-out streams, 20 % audits): alearned exceeds ε in 9 % / 4 % of streams vs arisk's 40 % / 35 %, at
  more automation (0.267 vs 0.236; 0.372 vs 0.348).
- Caveat: the five features, and alearned's γ = 0.005, 0.8·ε target and doubly-robust feedback, were chosen after
  looking at the test splits. H4 or a fresh collection should confirm them.
- Finding for the fine-tune: on action-prediction steps zero-shot Kev ranks BLOCKED first 30 % of the time and is
  wrong on 239 of those 242 steps.

How to use: `--policy learned_pac:0.05` (run_miniwob), `--systems supervisor,executor,fixed,escalate_on_none,risk,arisk,learned,alearned`
(run_h4; `learned` = learned_pac). The six spec systems stay the default.

## Independent review (2026-11-24)

The fourth fix round of 2026-11-16 had never been reviewed, and today's gates were new. One independent agent
reviewed both, replaying streams and probing the code; it found eight defects, seven fixed in `2706e38` with
regression tests (349 passed, 2 skipped; browser tests 3 passed):

1. The live `alearned` system ran with γ = 0.02, not the tuned 0.005: `fit_gate`'s own default overrode it. Over ε in
   18.5 % of streams instead of 9 % at ε = 0.05. Fixed: each adaptive gate keeps its own default.
2. Unanswered audit replies lowered the label rate to r·q and biased arisk's live error estimate low (with 40 %
   unanswered: auto_err 0.049 → 0.072, streams over ε 43 % → 83 %). alearned's doubly-robust default was unaffected.
   Fixed: the H4 stream counts audits asked and answered; answered ones weigh 1/(r·q̂); the counts survive a resume.
   Still use `--supervisor-json` (H3 saw 37–58 % unparsed replies without it).
3. CamelCase splitting had broken credential stops for names like `passCode`, `CVVCode`, `OTPCode`, `cardNum`, and
   turned `UnSubscribe` into a payment stop and `postCode` into a send. Fixed: names are read as written, then as
   words. WebGuard numbers unchanged.
4. Month and week inputs got the datetime-local format hint. Fixed.
5. A failed episode fed its steps to the adaptive gate twice (once failing, once on retry). Fixed: α, the barrier
   count, both RNGs and the audit counts are restored before the retry.
6. A dead browser was recorded as an ordinary failed episode. Fixed: it is raised and the episode retried.
7. A nested control beat `<label for>`. Fixed (HTML's rule).
8. Not fixed, by design: the learned model's terminal-option feature was fitted on WebJev's DONE / BLOCKED, while
   live it fires on none_of_these, which WebJev never has. It errs toward escalating, so H4 may show the learned
   gates making more supervisor calls than offline. A live collection with none_of_these is the fix.

Verified correct by the same review: the CRC and LTT thresholds (brute force on 3,000 tied cases), `binom_cdf`, tie
handling, one-class and tiny calibration sets, fold-model averaging, the doubly-robust estimate on every runner path,
resume (10 interrupt points, two adaptive systems, exact match), no calibration/test leakage, and the barrier bound
when every audit is answered.

## Research

`docs/research/2026-11-24-improvements.md`:
- §1 where the headroom is;
- §2 selective prediction, scores and PAC guarantees;
- §3 online risk control with audited feedback;
- §4 web-agent safety (detectors, production confirmation policies, prompt injection);
- §5 what was built and measured;
- §6 ranked roadmap.

Two results changed the plan:
- **Exceedance is structural** (§3). Any controller aimed exactly at ε exceeds it on roughly half of finite streams,
  because the 20 % audits make the error signal noisy (SD ≈ 0.018 at ε = 0.05, T = 600). Fixes:
  - a margin;
  - a higher audit rate;
  - audits targeted by predicted risk;
  - for a formal per-stream bound, an audited-error barrier. On M1's streams it costs a third of the automation at a
    20 % audit rate and almost nothing at 50 %.
- **Surety's keyword detector is far below even zero-shot LLMs** on WebGuard (16 % vs 26–59 % HIGH recall) (§4).
  Production agents confirm state-changing actions with ~92–100 % recall. The recommended shape is two tiers: broad
  keywords, then an LLM judge on the flagged steps only. Hard stops are never lifted by the supervisor; Surety already
  does this.

## State of the tree

- Branch `overnight-2026-11-16`: the rebuilt 2026-11-16 work plus today's commits on top of `006f180`; pushed to
  `origin`, `main` untouched. A verified bundle of `main` + the branch is kept locally (see "What needs you").
- `uv run pytest -q`: 349 passed, 2 skipped; `SURETY_LIVE=1 uv run pytest tests/integration`: 3 passed (MiniWoB++
  served locally on :8765, stopped afterwards).
- Every table regenerates on CPU from `artifacts/m1/` (`offline_analyze`, `offline_ci`, `offline_sweep`,
  `offline_groups`, `offline_learned`, `eval_webguard`); no model was run, as before (no local models on this laptop).
- Untouched: a training job from another project (`p1`, SmolLM) was running on this laptop all night.

## Addendum 2026-11-25: correcting the executor's label bias

Asked "we have hit a ceiling; what next?": the ceiling is the executor (46 % right on held-out sites; a perfect gate
would automate 51 % at ε = 0.05, the PAC gate 27 %). The cheapest lever, an eight-number per-action prior correction
(`PriorCorrector`, `scripts/offline_prior.py`, `results/m1/kev_compact_prior.md`), was run through the full pipeline:

- Accuracy on held-out sites 0.458 → 0.588 (action prediction 0.310 → 0.570): Kev's raw scores are badly biased
  toward BLOCKED and away from CLICK.
- On calibration sites the PAC gate then automates 0.306 / 0.453 at ε = 0.05 / 0.1 (from 0.273 / 0.388) within ε.
- On held-out sites it overshoots (0.052 and 0.116, over ε in 49 % / 92 % of 200 draws), entirely through
  action-prediction steps, whose errors rank worse on new sites; two fixes tried did not close the gap. Adaptive
  gates hold ε on held-out streams but keep little of the gain (automation 0.266 / 0.406 vs 0.267 / 0.372).
- Not adopted as a default. It sharpens the plan: the fine-tune should target the label bias, and calibration data
  must come from more sites for accuracy gains to become automation that holds on new sites.

Also: on GitHub, `main` and this branch were rewritten overnight (identical files; the early commits' author email
changed). The local branches now follow the rewritten history, and new commits build on it.

## Addendum 2026-11-25 (later): fine-tune prep

The owner chose the Kev-4B fine-tune as the next step and is setting up Modal. Ready, not run:

- **Plan M2** (`artifacts/m2/plan.json`, `make_plan.py --design spread`): training, calibration and test websites are
  disjoint (610 / 304 / 304). Hosts are grouped into websites by registrable domain (`surety.eval.sites.website`:
  en. and de.wikipedia.org, apple.com and apps.apple.com, amazon.com and amazon.de are one website). Each split is
  spread over its websites (at most 4 steps a website; cal spans 283 websites, test_shift 304) and alternates task
  types. On M1's plan a fine-tuned Kev would be calibrated on its own training sites (549 of 600 cal steps), and the
  prior-correction result shows that calibration on one set of sites over-promises on others.
- **Training data** rebuilt from M2's training websites (`artifacts/m2/kev_ft_stats.json`): 8,000 / 500 steps from
  529 / 31 websites; no calibration or test step is on a training website. BLOCKED is the gold answer on 1.25 % of
  its action steps. The training launcher refuses data not built from M2's plan.
- **Scoring job** `scripts/modal_score_kev.py`: kev.serve and offline_collect in one container on the `surety-kev-ft`
  volume, so a checkpoint is scored where it was trained. Rows, model card and log are saved to `runs/scores/<name>/`
  (`--fetch` after a disconnect). It refuses a server that runs anything but the requested checkpoint (kev.serve
  otherwise falls back silently) and a `--name` already scored with another checkpoint or plan. Tested locally
  against a stand-in server (`tests/test_serve_collect.py`); both Modal launchers load and parse their options with
  modal 1.5.5; nothing has run on Modal.
- **Parity check** `scripts/score_parity.py`: re-score 100 of M1's steps zero-shot first and compare with M1's rows.
- **Review** (independent agent, of the first version): seven findings, all fixed. The plan split hosts, not
  websites: 41 websites spanned groups, and 221 of its 1,600 cal/test steps were on a website the training file
  trains on. The documented `modal volume get` would have misplaced the rows. Re-using a `--name` with another
  checkpoint would have passed old rows as new. cal and test_in were grouped by task type, so a `--limit` skewed.
  Nothing tied the uploaded training file to M2. Long runs lacked `--detach`. The fine-tuned head is served at
  temperature 1.0; the gates refit T on cal, so this is documented, not changed.
- **M1 caveat** found on the way: M1 also holds out hosts, not websites. 89 of its 600 test_shift steps (15 %) are on a
  website that also has calibration hosts, so M1's held-out-site numbers are somewhat optimistic (report §6). M1's
  artifacts are unchanged.

Order once Modal works: smoke train, parity, full train, zero-shot and fine-tuned re-scores on M2, then the offline
analyses on M2's rows. `uv run pytest -q`: 379 passed, 2 skipped.
