# Research: how to make Surety better (2026-11-24)

Three literature scans (online risk control; selective deferral and confidence scores; web-agent safety), each
checked against M1's numbers, plus what was built and measured from them the same day. Citations were opened on
arXiv or the venue page by the research agents; anything they could not open is marked UNVERIFIED.

## 1 Where the headroom is (M1 numbers)

- The executor is the ceiling. Kev-4B zero-shot is right on 47 % of steps, so a perfect ranker would automate
  0.47 + ε (52 % at ε = 0.05) and a random one ε / 0.53 (9 %). Risk control automated 22 %: it handled only about a
  third of the steps the executor already gets right. Ranking (the confidence score) is therefore the main lever
  for the gate; accuracy (the fine-tune) is the main lever overall.
- ε bounds the joint rate P(act ∧ wrong), not the error among acted steps: at ε = 0.05 and 22 % automation up to
  ~23 % of autonomous actions are wrong (0.05 / 0.22). If the owner wants the error *among acted steps* bounded
  instead, the selective-risk variants below (SGR, Trust or Escalate) control that, at a cost of more slack.
- Going from ε = 0.05 to 0.10 admits ~7 more points of steps, of which ~70 % are wrong: the extra automation is
  mostly bought with errors.

## 2 Selective prediction, scores and PAC guarantees

**Recommendations (ranked).**

1. A learned error predictor with one global threshold: the largest expected gain [13, 6, 19]. **Built**
   (`LearnedRisk`, §5).
2. A Learn-then-Test (PAC) threshold: P(auto_err > ε) ≤ δ instead of E[auto_err] ≤ ε, at ~2–5 points of automation
   at ε = 0.05 with 600 calibration rows (CRC allows up to 29 autonomous errors on 600 rows at ε = 0.05; the exact
   binomial test at δ = 0.1, 22; Hoeffding–Bentkus, 20; Hoeffding, 3) [1]. **Built** (`learned_pac`).
3. Task type and log(option count) as score features rather than per-group thresholds: group-conditional CRC cost
   22–37 points of automation in [9], and M1's per-type gates failed under site shift. **Built** (log option count
   and a terminal-option flag are features).
4. Agreement across option orderings (share of orderings whose top pick is the final pick, worst-case margin):
   free, no data split [4, 15]. Tried as extra features: no gain on M1 (§5).
5. "None of these" as a feature, or escalate when it wins: adding that option hurts calibration [14]. The terminal
   flag (DONE / BLOCKED / none_of_these) is the second-strongest feature in the learned model.
6. Later: a probe on the executor's hidden states, the most stable family for open-weight GUI grounding in [19]
   (needs hidden states logged).

**Procedures.**

- LTT [1]: fix a threshold grid in advance, ordered strict → permissive; p_j = P(Bin(n, ε) ≤ n·R̂_j) (binomial, exact
  for a 0/1 loss); reject while p_j ≤ δ and stop at the first failure; deploy the most permissive rejected
  threshold. Picking the most permissive threshold whose bound is ≤ ε *without* stopping at the first failure loses
  family-wise control.
- SGR [3]: binary search on a confidence threshold, binomial-tail bound on the error *among accepted* rows at
  δ / ⌈log₂ m⌉ per step.
- Trust or Escalate [4]: the same selective-risk bound, scanned from the strictest threshold; for a cascade, each
  stage is calibrated on the rows the earlier stages escalated, at δ / (number of stages).
- Conformal abstention [5] applies CRC to 1{answer ∧ wrong}; conformal alignment [6] fits a logistic error model,
  then conformal p-values with Benjamini–Hochberg (FDR in expectation, batches); cross-validation CRC [8] reuses
  folds with an expectation-only guarantee.

**Evidence on scores.** Max probability is a strong baseline [3]; option order changes LLM multiple-choice accuracy
by 13–85 % [12] and averaging over permutations adds 4.9–7.9 points [11]; consistency ensembles raised AUROC from
0.642 to 0.723 [4]; a random forest on top-5 probabilities raised coverage at 80 % accuracy from 48 % to 56 % [13];
a few hundred labelled rows suffice for a learned scorer [6].

**Agents that ask for help.** KnowNo [15] (act only on a singleton conformal set; help requests −8–24 %);
introspective planning [17] (better scores: help requests 77.5 % → 63 %, unnecessary asking 51 % → 6 %); OS-Kairos
[16] (learned per-step confidence for GUI agents, no guarantee); SafeGround [18] (Clopper–Pearson threshold,
escalate to a stronger model: +5.4 points); Argus [19] (conformal sets under-cover when the interface shifts);
CURA [20] (certified trajectory alarms; its composite score did not beat a token-count baseline).

**Pitfalls that apply to Surety.**

- Every variant compared on the test rows is another test. The learned gate's feature set was chosen after an
  exploratory look at test AUROC (9 candidate features), and alearned's step size and margin were picked on the
  test_shift streams: both are in-sample choices, stated as such in the results.
- T is fitted on the calibration rows, as are the gates; a cleaner protocol fits T and the scorer on separate rows
  (e.g. the in-site test rows) and calibrates the threshold on the rest.
- Sites, not rows, are the exchangeable units: the effective sample is smaller than 600, a shift in group mix broke
  CRC in up to 47 % of trials in [9], and a per-site guarantee needs hierarchical conformal methods [21].
- A per-step guarantee is not a per-episode guarantee.

### References for §1–2

[1] Angelopoulos, Bates, Candès, Jordan, Lei. Learn then Test: Calibrating Predictive Algorithms to Achieve Risk
Control. Annals of Applied Statistics, 2025. https://arxiv.org/abs/2110.01052
[2] Angelopoulos, Bates, Fisch, Lei, Schuster. Conformal Risk Control. ICLR 2024. https://arxiv.org/abs/2208.02814
[3] Geifman, El-Yaniv. Selective Classification for Deep Neural Networks. NIPS 2017. https://arxiv.org/abs/1705.08500
[4] Jung, Brahman, Choi. Trust or Escalate: LLM Judges with Provable Guarantees for Human Agreement. ICLR 2025.
https://arxiv.org/abs/2407.18370
[5] Abbasi Yadkori et al. Mitigating LLM Hallucinations via Conformal Abstention. arXiv, 2024.
https://arxiv.org/abs/2405.01563
[6] Gui, Jin, Ren. Conformal Alignment: Knowing When to Trust Foundation Models with Guarantees. NeurIPS 2024.
https://arxiv.org/abs/2405.10301
[7] Pona et al. Calibrate-Then-Delegate: Safety Monitoring with Risk and Budget Guarantees via Model Cascades.
arXiv, 2026. https://arxiv.org/abs/2604.14251
[8] Cohen, Park, Simeone, Shamai. Cross-Validation Conformal Risk Control. ISIT 2024. https://arxiv.org/abs/2401.11974
[9] Salem et al. Hierarchical Group-Conditional Conformal Risk Control for Selective Prediction in Language Models.
arXiv, 2026. https://arxiv.org/abs/2607.24562
[10] Xu, Guo, Wei. Selective Conformal Risk Control. arXiv, 2025. https://arxiv.org/abs/2512.12844
[11] Zheng et al. Large Language Models Are Not Robust Multiple Choice Selectors. ICLR 2024.
https://arxiv.org/abs/2309.03882
[12] Pezeshkpour, Hruschka. Large Language Models Sensitivity to The Order of Options in Multiple-Choice Questions.
Findings of NAACL 2024. https://aclanthology.org/2024.findings-naacl.130
[13] Kamath, Jia, Liang. Selective Question Answering under Domain Shift. ACL 2020. https://arxiv.org/abs/2006.09462
[14] Kadavath et al. Language Models (Mostly) Know What They Know. arXiv, 2022. https://arxiv.org/abs/2207.05221
[15] Ren et al. Robots That Ask For Help: Uncertainty Alignment for Large Language Model Planners. CoRL 2023.
https://arxiv.org/abs/2307.01928
[16] Cheng et al. OS-Kairos: Adaptive Interaction for MLLM-Powered GUI Agents. Findings of ACL 2025.
https://arxiv.org/abs/2503.16465
[17] Liang, Zhang, Fisac. Introspective Planning: Aligning Robots' Uncertainty with Inherent Task Ambiguity. NeurIPS
2024. https://arxiv.org/abs/2402.06529
[18] Wang, Fan, Wang. SafeGround: Know When to Trust GUI Grounding Models via Uncertainty Calibration. arXiv, 2026.
https://arxiv.org/abs/2602.02419
[19] Kumar et al. Uncertainty Quantification for Computer-Use Agents: A Benchmark across Vision-Language Models and
GUI Grounding Datasets. NeurIPS 2026 (accepted, per arXiv). https://arxiv.org/abs/2606.25760
[20] Kumar et al. CURA: Certified Runtime Alarms for Computer-Use Agents. arXiv, 2026. https://arxiv.org/abs/2608.27808
[21] Dunn, Wasserman, Ramdas. Distribution-Free Prediction Sets for Two-Layer Hierarchical Models. JASA, 2023.
https://arxiv.org/abs/1809.07441

## 3 Online risk control with audited feedback

**Why arisk exceeds ε in 40 % of streams.** For any ACI-style update α ← α + γ(ε − f_t), with f_t = A_t·ℓ_t / r and
audit coin A_t ~ Bern(r), the unclipped identity (gradient equilibrium [O7]) for S_T, the count of acted-and-wrong
steps, is

  S_T − εT = (α₁ − α_{T+1}) / γ − Σ_t (A_t / r − 1)·ℓ_t.

The first term is the cost of walking α down onto hard sites (present even at r = 1); the second is audit noise, with
SD ≈ √(εT(1 − r)/r): 0.018 (ε = 0.05) and 0.026 (ε = 0.1) in rate at T = 600, r = 0.2. Any controller centred on ε
therefore exceeds it in roughly 40–50 % of finite streams; no online algorithm removes this alone. Rolling risk control
[O6] has a two-sided guarantee only when the loss is below target on the full set and above it on the empty set;
Surety's loss is 0 on both (escalating everything is error-free), so nothing pushes α back up a wrong path except
audits, hence the drift to α = 0.5 at r = 0. Methods that are exact for the long-run average (ACI [O1], DtACI [O2],
conformal PID [O3], decaying steps [O4], SAOCP [O5], betting ACI [O8]) do not fix finite-stream exceedance: a toy
replication matched arisk (mean 0.048, 40 % over at ε = 0.05, r = 0.2), stayed 40 % over at r = 1, DtACI with IPW
expert losses was 83 % over, and decaying steps gave no gain.

**Ranked recommendations.**

1. **A target margin** (ε' < ε) with a small step: cheap, no guarantee. *Built*: alearned's 0.8·ε target with γ =
   0.005 exceeds ε in 9 % / 4 % of M1's held-out streams (§5). A principled size solves x + z·√(x(1 − r)/r) = εT for
   ε' = x/T (z = 1.645): 0.028 / 0.066 at ε = 0.05 / 0.1, r = 0.2, T = 600, i.e. a larger margin than 0.8·ε.
2. **An audited-error barrier (finite-horizon, per stream).** With error budget B = ⌊εT⌋ and k* = max{k :
   BinomCDF(k; B, r) ≤ δ}, stop acting once more than k* audited errors have been seen. Then P(S_T > εT) ≤ δ for any
   stream, executor and site order: by optional skipping, the audit coins on error steps are i.i.d. Bern(r), so a
   (B + 1)-th error needs at most k* audits among the first B. B = 30 (ε = 0.05, T = 600) gives k* = 2 at r = 0.2.
   Derived by the research agent, unpublished: needs checking. Toy cost at r = 0.2: mean error 0.023, 2.5 % of streams
   over ε, automation 0.43 vs 0.70 for plain ACI; much cheaper at r = 0.5 (0.59 vs 0.73). For an unknown horizon,
   Ville's inequality [O18] on the audited-error likelihood ratio gives an anytime version, looser still.
3. **Raise r, or audit by predicted risk with a floor** (active, anytime-valid RCPS [O9]): the required margin scales
   as √((1 − r)/r), so targeting audits at steps the error model doubts buys the most per supervisor call. Needs the
   runner to take a per-step audit probability.
4. Keep ACI / rolling RC on importance-weighted feedback as the inner controller, γ ≈ 0.005–0.02 [O6, O1].
5. Decaying steps reset at site boundaries [O4]: smoother α, no exceedance gain in the toy.

**Partial feedback.** Labels only on abstention force a trade-off of about T^μ mistakes against T^{1−μ} extra
abstentions, and that is tight [O11]; query-or-predict reaches coverage ≥ β − O(T^{−1/3}) with ~T^{−1/3} queries [O12]:
exploration audits are necessary, so r = 0 must fail, as M1 found. Importance weighting stays unbiased but grows the
second moment by 1/r [O17]; with known propensities (1 when escalated, r when acted) every candidate α's loss is
IPW-estimable, which enables expert or grid methods. A doubly-robust estimate [O9] cuts variance by about (1 − q) when
errors are rare: modest, as M1's streams showed (§5).

### References for §3

[O1] Gibbs, Candès. Adaptive Conformal Inference Under Distribution Shift. NeurIPS 2021. https://arxiv.org/abs/2106.00170
[O2] Gibbs, Candès. Conformal Inference for Online Prediction with Arbitrary Distribution Shifts. JMLR 2024.
https://arxiv.org/abs/2208.08401
[O3] Angelopoulos, Candès, Tibshirani. Conformal PID Control for Time Series Prediction. NeurIPS 2023.
https://arxiv.org/abs/2307.16895
[O4] Angelopoulos, Barber, Bates. Online conformal prediction with decaying step sizes. ICML 2024.
https://arxiv.org/abs/2402.01139
[O5] Bhatnagar, Wang, Xiong, Bai. Improved Online Conformal Prediction via Strongly Adaptive Online Learning. ICML 2023.
https://arxiv.org/abs/2302.07869
[O6] Feldman, Ringel, Bates, Romano. Achieving Risk Control in Online Learning Settings. TMLR 2023.
https://arxiv.org/abs/2205.09095
[O7] Angelopoulos, Jordan, Tibshirani. Gradient Equilibrium in Online Learning: Theory and Applications. JMLR 2025.
https://arxiv.org/abs/2501.08330
[O8] Podkopaev, Xu, Lee. Adaptive Conformal Inference by Betting. ICML 2024. https://arxiv.org/abs/2412.19318
[O9] Xu, Karampatziakis, Mineiro. Active, anytime-valid risk controlling prediction sets. NeurIPS 2024.
https://arxiv.org/abs/2406.10490
[O10] Khosravi, Huo. Conformal Selective Acting: Anytime-Valid Risk Control for RLVR-Trained LLMs. Preprint 2026.
https://arxiv.org/abs/2605.20270
[O11] Gangrade, Kag, Cutkosky, Saligrama. Online Selective Classification with Limited Feedback. NeurIPS 2021.
https://arxiv.org/abs/2110.14243
[O12] Skalse, Pona, Simeone, Paoletti. Online Conformal Prediction Beyond Feedback. Preprint 2026.
https://arxiv.org/abs/2608.07139
[O13] Wang, Simeone. Capacity-Constrained Delayed Feedback for Online Conformal Risk Control. Preprint 2026.
https://arxiv.org/abs/2611.17059
[O14] Ge, Bastani, Bastani. Stochastic Online Conformal Prediction with Semi-Bandit Feedback. ICML 2025.
https://arxiv.org/abs/2405.13268
[O15] Yang, Kim, Park. Online Conformal Prediction with Adversarial Semi-bandit Feedback via Regret Minimization. ICLR
2026. https://arxiv.org/abs/2604.17984
[O16] Lee, Jung, Park. Online Conformal Abstention Under Adversarial Bandit Feedback. NeurIPS 2026 (per arXiv).
https://arxiv.org/abs/2506.14067
[O17] Cesa-Bianchi, Lugosi, Stoltz. Minimizing Regret with Label Efficient Prediction. COLT 2004.
https://mlanthology.org/colt/2004/cesabianchi2004colt-minimizing
[O18] Howard, Ramdas, McAuliffe, Sekhon. Time-uniform Chernoff bounds via nonnegative supermartingales. Probability
Surveys 2020. https://arxiv.org/abs/1808.03204
[O19] Aldirawi, Li, Guo. Conformal Risk Control under Non-Monotone Losses: Theory and Finite-Sample Guarantees.
Preprint 2026. https://arxiv.org/abs/2604.01502 (Surety's RiskConformal controls a monotone envelope of its loss,
and the learned gates' loss is monotone, so this failure mode does not apply.)

## 4 Web-agent safety: detectors, confirmations, injection

**What WebGuard itself found [S1].** Zero-shot frontier models reach 26–59 % HIGH-risk recall from text (accessibility
tree + HTML) and 24–51 % from screenshots (GPT-4o-mini reaches 64–80 % only by calling almost everything HIGH); best
zero-shot accuracy is 69 %. A fine-tuned Qwen2.5-VL-7B reaches 75–88 % accuracy and 67–90 % HIGH recall. Models lean
on the button label (a "Checkout" that only advances the flow gets called HIGH) and over-flag intermediate steps. The
paper reports no SAFE false-positive rate, so Surety's 2.6 % (labels) / 5.8 % (label + URL) has no published
comparison. Surety's keyword detector (16 % / 24 % HIGH recall) is far below even zero-shot LLMs.

**Benchmarks and guardrails (2024–26).**
- Benchmarks:
  - ST-WebAgentBench [S2]: policy-compliant completion 15.0 % vs 24.3 % raw.
  - SafeArena [S3]: GPT-4o completed 34.7 % of harmful tasks.
  - OS-Harm [S4] and RiOSWorld [S5].
  - Agent-SafetyBench [S6]: no agent above 60 % safe.
  - MobileSafetyBench [S7].
  - WASP [S8]: hijacks start in up to 86 % of runs and succeed in up to 17 %.
  - AgentDojo [S9], GPT-4o: attack success 57.7 % undefended, 6.8 % with a tool filter, 8.0 % with a detector that
    cut utility from 69 % to 41 %.
- Guardrails:
  - GuardAgent [S10], AGrail [S11], ShieldAgent [S12] and Conseca [S13].
  - CaMeL [S14]: 77 % of tasks with provable security vs 84 % undefended.
  - Progent [S15]: attack success 39.9 % → 0–1 %, utility 79.4 % → 76.3 %.

**How production agents confirm.**
- Operator [S16] confirms state-changing actions (92 % recall), hands credentials and payment to the user (takeover
  mode) and watches sensitive sites.
- ChatGPT agent [S17] reports 99.9–100 % confirmation recall on permissions, high-stakes messages and payments.
- OpenAI's guide [S18] treats typing sensitive data into a form as data transmission and says on-screen text never
  grants permission.
- Anthropic [S19, S20] keeps a human on consequential actions and consent (cookies, payments, terms) and asks before
  publishing, purchasing or sharing personal data.
- Google's computer-use safety service [S22] checks every step and returns `require_confirmation` or block for
  financial, consent, communication, legal and data actions.

**Recommendations for Surety (ranked; where Surety already stands).**

1. **Stop credential and card typing on field attributes, not labels.** Use `type=password`, autocomplete values
   current-password / new-password / one-time-code / cc-*, and `cvv|otp|iban` in the field name or id [S23]. *Mostly
   done*: both envs read input type and autocomplete from the DOM; IBAN and routing-number stems are not in the
   credential list.
2. **Never let the supervisor lift a hard stop; page text may only raise risk.** LLM monitors can be hijacked: a
   GPT-4.1 monitor rated all 375 injected backdoors benign [S25], and adaptive attacks beat most detectors and
   spotlighting [S26]. *Done*: the payment and credential stops apply to the final action, whoever chose it.
3. **A two-tier detector.** *Owner's decision*; it fits the existing escalation path, since non-payment critical
   categories already escalate.
   - First tier: keywords plus the WebGuard-motivated families (submit, sign up, save, download, non-English verbs).
     Cheap, with HIGH recall ~0.66 on label + URL.
   - Second tier: an LLM judge on the flagged steps only, with WebGuard's rubric (reversible? who is affected?) and a
     fixed label as output.
   - Cascades cut cost ~40× [S27], and combining monitors doubled recall at a fixed budget [S28].
   - The judge's job is to clear generic "submit" flags (the owner's objection to the families), not to catch misses:
     total recall cannot exceed the first tier's.
4. **Commit-step features.** A commit verb plus a confirm / review / payment token in the URL → HIGH; flow verbs
   (continue, checkout, add to cart) → LOW [S1, S2]. Testable offline on WebGuard (label + URL).
5. **Structural priors.** `type=submit`, POST forms (GET is "safe" by RFC 9110 [S29]), account and checkout paths.
   Partly testable on WebGuard through its `tagHead` field; no published test found.
6. **Categories production agents gate and Surety does not:** terms and consent, subscriptions, permissions and API
   keys, downloads, CAPTCHAs [S18, S22].
7. **Supervisor input/output hardening.**
   - Mark page text as data: datamarking cut static attacks from > 50 % to < 2 %; adaptive ones still get through
     [S26, S30].
   - Keep trusted inputs apart from page text (CaMeL-style [S14]).
   - Allow only option ids or fixed labels as output [S31]; Surety's `--json` mode already constrains the id.
   - Never resample.
8. **Single-use approvals bound to the exact element, value and URL**, enforced by the executor: cut irreversible
   harm from 37.5–60 % to 2.5 % in GateArena [S24]. Needs live tests.

What can be evaluated on WebGuard without training:
- Offline: 3 (the first tier; the judge with an API key), 4, and multilingual lists grouped by site.
- Partly, through `tagHead`: 1, 5 and 6.
- Not offline: 2, 7 and 8.
- Keep each site in one split.

### References for §4

[S1] Zheng et al. WebGuard. 2025. https://arxiv.org/abs/2507.14293
[S2] Levy et al. ST-WebAgentBench. ICLR 2026. https://arxiv.org/abs/2410.06703
[S3] Tur et al. SafeArena. ICML 2025. https://arxiv.org/abs/2503.04957
[S4] Kuntz et al. OS-Harm. NeurIPS 2025 D&B. https://arxiv.org/abs/2506.14866
[S5] Yang et al. RiOSWorld. NeurIPS 2025. https://arxiv.org/abs/2506.00618
[S6] Zhang et al. Agent-SafetyBench. 2024. https://arxiv.org/abs/2412.14470
[S7] Lee et al. MobileSafetyBench. 2024 (venue UNVERIFIED). https://arxiv.org/abs/2410.17520
[S8] Evtimov et al. WASP. NeurIPS 2025. https://arxiv.org/abs/2504.18575
[S9] Debenedetti et al. AgentDojo. NeurIPS 2024. https://arxiv.org/abs/2406.13352
[S10] Xiang et al. GuardAgent. ICML 2025. https://arxiv.org/abs/2406.09187
[S11] Luo et al. AGrail. ACL 2025. https://aclanthology.org/2025.acl-long.399
[S12] Chen et al. ShieldAgent. ICML 2025. https://arxiv.org/abs/2503.22738
[S13] Tsai, Bagdasarian. Contextual Agent Security (Conseca). HotOS 2025. https://arxiv.org/abs/2501.17070
[S14] Debenedetti et al. Defeating Prompt Injections by Design (CaMeL). 2025. https://arxiv.org/abs/2503.18813
[S15] Shi et al. Progent. 2025. https://arxiv.org/abs/2504.11703
[S16] OpenAI. Operator System Card. 2025. https://openai.com/index/operator-system-card
[S17] OpenAI. ChatGPT agent System Card. 2025. https://deploymentsafety.openai.com/chatgpt-agent/user-confirmations
[S18] OpenAI. Computer use guide. https://developers.openai.com/api/docs/guides/tools-computer-use-integration
[S19] Anthropic. Computer use tool. https://platform.claude.com/docs/en/agents-and-tools/computer-use
[S20] Anthropic. Piloting Claude in Chrome. 2025. https://claude.com/blog/claude-for-chrome
[S21] Google. Gemini 2.0 (Project Mariner). 2024.
https://blog.google/technology/google-deepmind/google-gemini-ai-update-december-2024/
[S22] Google. Computer Use. https://aistudio.google.com/docs/computer-use
[S23] WHATWG. HTML Standard: autofill. https://html.spec.whatwg.org/multipage/form-control-infrastructure.html#autofill
[S24] Jiang et al. GateArena. NeurIPS 2026 workshop. https://neurips.cc/virtual/2026/168290
[S25] Terekhov et al. Adaptive Attacks on Trusted Monitors. ICLR 2026. https://arxiv.org/abs/2510.09462
[S26] Nasr et al. The Attacker Moves Second. USENIX Security 2026. https://arxiv.org/abs/2510.09023
[S27] Cunningham et al. Constitutional Classifiers++. ICLR 2026. https://arxiv.org/abs/2601.04603
[S28] Hua et al. Combining Cost-Constrained Runtime Monitors. NeurIPS 2025. https://arxiv.org/abs/2507.15886
[S29] IETF. RFC 9110, safe methods. 2022. https://www.rfc-editor.org/rfc/rfc9110.html#name-safe-methods
[S30] Hines et al. Spotlighting. CAMLIS 2024. https://arxiv.org/abs/2403.14720
[S31] Beurer-Kellner et al. Design Patterns for Securing LLM Agents. 2025. https://arxiv.org/abs/2506.08837

## 5 What was built and measured (2026-11-24)

From §2's first three recommendations: `src/surety/gate/learned.py` (`LearnedRisk`, `learned_pac`,
`AdaptiveLearnedRisk`), `scripts/offline_learned.py`, results in `results/m1/kev_compact_learned.md` and
`docs/results/m1.md`.

- **Ranking.** AUROC for "top option wrong" on held-out sites: p_top 0.697, margin 0.752, learned 0.838.
- **Features tried.** Nine candidates were tried on an exploratory look:
  - The final five come from the probability vector, so the gate runs live unchanged: p_top, margin, normalised
    entropy, log option count, top is terminal.
  - Permutation agreement, the minimum per-ordering top probability and task type gave no further automation.
- **Calibration draws** (200, held-out sites), mean auto_err (share of draws over ε) / automation:

  | ε | risk | learned_pac | learned |
  |---|---|---|---|
  | 0.05 | 0.037 (0.000) / 0.226 | 0.038 (0.005) / 0.274 | 0.046 (0.130) / 0.302 |
  | 0.1 | 0.054 (0.000) / 0.294 | 0.078 (0.020) / 0.379 | 0.095 (0.260) / 0.411 |

- **Per task type** at ε = 0.05: learned_pac holds grounding at 0.039 (risk: 0.059, over ε in 70 % of draws) and
  doubles action-prediction automation (0.160 vs 0.083).
- **Validity check.** Fitting T and the model on 400 separate rows and calibrating on the 600 cal rows (exact
  validity) gives the same numbers as the shipped cross-fitting (60 draws).
- **Online** (200 streams, audit rate 0.2). alearned (γ 0.005, target 0.8·ε, doubly-robust feedback) exceeds ε in
  9 % / 4 % of streams vs arisk's 40 % / 35 %, at more automation (0.267 vs 0.236; 0.372 vs 0.348). These settings
  were picked on the same streams (24 simulated configurations): in-sample.
- **A failure mode worth fixing in the fine-tune.** On action-prediction steps zero-shot Kev ranks BLOCKED first on
  30 % of steps and is wrong on 239 of those 242; DONE first on 150 and wrong on 95.

- **Per-stream guarantees on M1's 200 held-out streams** (`results/m1/kev_compact_learned.md`; the r = 0.5 rows from
  a scratch simulator of the same streams):

  | ε, r | config | auto_err | streams over ε | automation |
  |---|---|---|---|---|
  | 0.05, 0.2 | arisk (M1) | 0.048 | 40 % | 0.236 |
  | 0.05, 0.2 | default (0.8·ε target) | 0.036 | 9 % | 0.267 |
  | 0.05, 0.2 | noise-sized margin (target 0.028) | 0.026 | 0.5 % | 0.227 |
  | 0.05, 0.2 | default + barrier (k* = 2) | 0.024 | 5 % | 0.165 |
  | 0.05, 0.5 | default + barrier (k* = 10) | 0.035 | 0.5 % | 0.255 |
  | 0.1, 0.2 | arisk (M1) | 0.094 | 35 % | 0.348 |
  | 0.1, 0.2 | default | 0.075 | 4 % | 0.372 |
  | 0.1, 0.2 | noise-sized margin (target 0.066) | 0.060 | 0 % | 0.340 |
  | 0.1, 0.2 | default + barrier (k* = 7) | 0.064 | 2.5 % | 0.309 |

  - At r = 0.2 the barrier's formal guarantee costs about a third of the automation.
  - The noise-sized margin keeps arisk's automation at near-zero exceedance, without a formal guarantee.
  - At r = 0.5 the barrier is almost free.

## 6 Roadmap (ranked)

1. **Fine-tune Kev-4B** (Modal launcher ready; owner's account). The executor's accuracy (0.47) caps everything above,
   and the BLOCKED failure mode is a clear target for training data. *Tried the cheap version on 2026-11-25*: eight
   per-action log-prior offsets fitted on cal raise held-out accuracy 0.458 → 0.588, but the gates cannot certify the
   extra action-prediction picks on new sites (static gates overshoot there; adaptive gates hold ε and keep little
   of the gain). See `docs/results/m1.md` and `results/m1/kev_compact_prior.md`. Calibration data from more sites
   is the other half of this item: an accuracy gain only becomes safe automation if the gate's calibration
   transfers to the sites it is deployed on. *Prepared 2026-11-25 (not run):* plan M2 with site-disjoint training,
   calibration and test sites, each split spread over its sites; training data from M2's training sites; a Modal job
   that scores a checkpoint where it was trained (handover 2026-11-24, last addendum).
2. **H4 with the learned gates.** Run `--systems ...,learned,alearned` next to arisk. This confirms on new data the
   choices made in-sample here: features, γ, margin.
3. **Two-tier critical detector.** Cheap high-recall keywords, then an LLM judge on the flagged steps (§4.3).
   - Add IBAN / routing-number credential stems.
   - Add the consent, subscription, permission / API-key and download categories.
   - Owner's decision.
4. **Audit by predicted risk with a floor** (§3.3). It is the cheapest way to shrink the margin. Needs a per-step
   audit probability in the runner.
5. **An optional barrier mode** for deployments that need a per-stream bound with a known horizon, with a higher
   audit rate.
6. **H3 with a stronger supervisor and `--json`** (owner's key). Add supervisor hardening: datamark page text, never
   resample (§4.7).
7. **Log per-ordering probabilities and executor hidden states live.** This enables agreement features and probes
   [19].
8. **If the owner prefers bounding the error among acted steps** (one wrong action in N) over the joint rate: a
   selective-risk variant (SGR [3] / Trust or Escalate [4]).
