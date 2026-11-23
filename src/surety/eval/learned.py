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
                    p, g = _split(rs, probs, split)
                    sets = _sets(policy, p)
                    out.append({"seed": seed, "eps": eps, "policy": policy.name, "split": split,
                                "auto_err": m.autonomous_error_rate(sets, g), "automation": m.automation_rate(sets)})
    return out


def draws_table(records: list[dict]) -> str:
    keys = list(dict.fromkeys((r["eps"], r["policy"], r["split"]) for r in records))
    body = []
    for eps, policy, split in sorted(keys, key=lambda k: (k[0], k[2], keys.index(k))):
        rs = [r for r in records if (r["eps"], r["policy"], r["split"]) == (eps, policy, split)]
        err, auto = np.array([r["auto_err"] for r in rs]), np.array([r["automation"] for r in rs])
        body.append([str(eps), split, policy, len(rs), float(err.mean()), _pct(err), float(np.mean(err > eps)),
                     float(auto.mean()), _pct(auto)])
    return _table(("eps", "split", "policy", "draws", "auto_err", "auto_err_2.5–97.5%", "P(auto_err>eps)",
                   "automation", "automation_2.5–97.5%"), body)


def stream_draws(rows: list[dict], eps: float, makers: dict, audit_rate: float = 0.2, n_seeds: int = 200) -> str:
    """Each adaptive gate on the same site-ordered test_shift streams and audit draws (as resample.adaptive_draws)."""
    probs = variants(rows)["debiased+T"]
    cal = _split(rows, probs, "cal")
    body = []
    for name, make in makers.items():
        err, tail, auto = [], [], []
        for seed in range(n_seeds):
            order = _stream_order(rows, seed)
            p, g = [probs[i] for i in order], [rows[i]["gold_id"] for i in order]
            policy = make(eps)
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


ADAPTIVE = {"arisk": lambda eps: AdaptiveRisk(eps), "alearned": lambda eps: AdaptiveLearnedRisk(eps)}
