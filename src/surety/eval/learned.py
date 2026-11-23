"""Learned error score vs top probability: ranking, calibration draws and adaptive streams (results/m1/*_learned.md)."""
import random

import numpy as np

from surety.calib import metrics as m
from surety.eval.analyze import TAIL, _sets, _split, _table, variants
from surety.eval.resample import _pct, _stream_order, resplit
from surety.gate import AdaptiveRisk, RiskConformal
from surety.gate.learned import AdaptiveLearnedRisk, LearnedRisk, features
from surety.gate.policies import _top

STATIC = (RiskConformal, LearnedRisk, lambda eps: LearnedRisk(eps, delta=0.1))
# risk keeps Figure 1's blue; aqua and yellow pass the CVD / normal-vision checks with it (all pairs, light surface)
# and sit below 3:1 contrast, so the figure names each line directly
SWEEP_COLORS = {"risk": "#2a78d6", "learned_pac": "#1baf7a", "learned": "#eda100"}
TESTS = ("test_in", "test_shift")


def auroc(score: np.ndarray, wrong: np.ndarray) -> float:
    """P(score of a wrong step > score of a correct step), ties counted half (Mann–Whitney)."""
    ranks = np.empty(len(score))
    order = np.argsort(score, kind="mergesort")
    ranks[order] = np.arange(1, len(score) + 1)
    for v in np.unique(score):  # average ranks over ties
        tie = score == v
        ranks[tie] = ranks[tie].mean()
    n_w, n_c = int(wrong.sum()), int(len(wrong) - wrong.sum())
    return float((ranks[wrong == 1].sum() - n_w * (n_w + 1) / 2) / (n_w * n_c)) if n_w and n_c else float("nan")


def ranking_table(rows: list[dict]) -> str:
    """How well each score ranks the executor's errors on the test splits (higher AUROC = better), by task type."""
    probs = variants(rows)["debiased+T"]
    gate = LearnedRisk(0.1)
    gate.fit(*_split(rows, probs, "cal"))
    names = ("p_top", "margin", "entropy")
    body = []
    for split in TESTS:
        for task in ("all", "action_prediction", "element_grounding"):
            idx = [i for i, r in enumerate(rows) if r["split"] == split and task in ("all", r["task_type"])]
            wrong = np.array([_top(probs[i])[0] != rows[i]["gold_id"] for i in idx], float)
            x = np.array([features(probs[i]) for i in idx])
            cols = [auroc(-x[:, 0], wrong), auroc(-x[:, 1], wrong), auroc(x[:, 2], wrong),
                    auroc(np.array([gate.score(probs[i]) for i in idx]), wrong)]
            body.append([split, task, len(idx), float(wrong.mean()), *map(float, cols)])
    return _table(("split", "task_type", "n", "error_rate", *(f"AUROC {n}" for n in names), "AUROC learned"), body)


def static_draws(rows: list[dict], eps_list=(0.05, 0.1), n_seeds: int = 200) -> list[dict]:
    out = []
    for seed in range(n_seeds):
        rs = resplit(rows, seed)
        probs = variants(rs)["debiased+T"]
        cal = _split(rs, probs, "cal")
        for eps in eps_list:
            for make in STATIC:
                policy = make(eps)
                if isinstance(policy, LearnedRisk):
                    policy.seed = seed
                policy.fit(*cal)
                for split in TESTS:
                    idx = [i for i, r in enumerate(rs) if r["split"] == split]
                    sets = _sets(policy, [probs[i] for i in idx])
                    for task in ("all", "action_prediction", "element_grounding"):
                        keep = [k for k, i in enumerate(idx) if task in ("all", rs[i]["task_type"])]
                        s, g = [sets[k] for k in keep], [rs[idx[k]]["gold_id"] for k in keep]
                        out.append({"seed": seed, "eps": eps, "policy": policy.name, "split": split, "task_type": task,
                                    "auto_err": m.autonomous_error_rate(s, g), "automation": m.automation_rate(s)})
    return out


def draws_table(records: list[dict], by_type: bool = False) -> str:
    """All steps (by_type=False), or each task type on its own (the guarantees are marginal: no per-type control)."""
    records = [r for r in records if (r.get("task_type", "all") != "all") == by_type]
    keys = list(dict.fromkeys((r["eps"], r["policy"], r["split"], r.get("task_type", "all")) for r in records))
    body = []
    for key in sorted(keys, key=lambda k: (k[0], k[2], k[3], keys.index(k))):
        rs = [r for r in records if (r["eps"], r["policy"], r["split"], r.get("task_type", "all")) == key]
        err, auto = np.array([r["auto_err"] for r in rs]), np.array([r["automation"] for r in rs])
        eps, policy, split, task = key
        body.append([str(eps), split, *([task] if by_type else []), policy, len(rs), float(err.mean()), _pct(err),
                     float(np.mean(err > eps)), float(auto.mean()), _pct(auto)])
    return _table(("eps", "split", *(["task_type"] if by_type else []), "policy", "draws", "auto_err",
                   "auto_err_2.5–97.5%", "P(auto_err>eps)", "automation", "automation_2.5–97.5%"), body)


def stream_draws(rows: list[dict], eps: float, makers: dict, audit_rate: float = 0.2, n_seeds: int = 200) -> str:
    """Each adaptive gate on the same site-ordered test_shift streams and audit draws (as resample.adaptive_draws).
    makers: name -> make(eps, horizon, audit_rate), horizon = the stream's length."""
    probs = variants(rows)["debiased+T"]
    cal = _split(rows, probs, "cal")
    body = []
    for name, make in makers.items():
        err, tail, auto = [], [], []
        for seed in range(n_seeds):
            order = _stream_order(rows, seed)
            p, g = [probs[i] for i in order], [rows[i]["gold_id"] for i in order]
            policy = make(eps, len(order), audit_rate)
            policy.fit(*cal)
            rng = random.Random(seed)
            sets = []
            for x, y in zip(p, g):
                c = policy.prediction_set(x)
                sets.append(c)
                w = 1.0 if len(c) != 1 else (1 / audit_rate if rng.random() < audit_rate else 0.0)
                policy.update(x, y, w)
            err.append(m.autonomous_error_rate(sets, g))
            tail.append(m.autonomous_error_rate(sets[-TAIL:], g[-TAIL:]))
            auto.append(m.automation_rate(sets))
        err, tail, auto = map(np.array, (err, tail, auto))
        body.append([name, f"{audit_rate:g}", n_seeds, float(err.mean()), _pct(err), float(np.mean(err > eps)),
                     float(tail.mean()), float(auto.mean()), _pct(auto)])
    return _table(("policy", "audit_rate", "streams", "auto_err", "auto_err_2.5–97.5%", "P(auto_err>eps)",
                   "tail_auto_err", "automation", "automation_2.5–97.5%"), body)


# arisk as in M1; the same score with alearned's step size and margin (target 0.8·eps); plain ACI on the learned score;
# then the defaults of AdaptiveLearnedRisk without and with the margin
ADAPTIVE = {
    "arisk (γ 0.02)": lambda eps, t, r: AdaptiveRisk(eps),
    "arisk (γ 0.005, target 0.8·eps)": lambda eps, t, r: AdaptiveRisk(0.8 * eps, gamma=0.005),
    "alearned (γ 0.02, κ 0, ipw)": lambda eps, t, r: AdaptiveLearnedRisk(eps, gamma=0.02, margin=0.0, feedback="ipw"),
    "alearned (γ 0.005, κ 0, dr)": lambda eps, t, r: AdaptiveLearnedRisk(eps, margin=0.0),
    "alearned (γ 0.005, κ 0.2, dr): default": lambda eps, t, r: AdaptiveLearnedRisk(eps),
    # research §3: a margin sized from the audit noise of a t-step stream, and the audited-error barrier (δ 0.1)
    "alearned, noise-sized margin": lambda eps, t, r: AdaptiveLearnedRisk(eps, margin="noise", horizon=t, audit_rate=r),
    "alearned default + barrier (δ 0.1)": lambda eps, t, r: AdaptiveLearnedRisk(eps, horizon=t, audit_rate=r,
                                                                               barrier_delta=0.1),
}
