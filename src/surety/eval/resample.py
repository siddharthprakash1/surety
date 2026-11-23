"""Uncertainty for the M1 gate tables.

draws: re-draw cal from the calibration-site pool (cal ∪ test_in) at its original size, refit T and the gates, score
test_in (the rest of the pool) and test_shift — the spread over calibration draws that the CRC guarantee averages over.
site_bootstrap: fit once on the original cal, resample test sites with replacement (rows within a site are correlated,
so a row-level Wilson interval is too narrow).
adaptive_draws: the online stream on test_shift under different site orders and audit draws.
"""
import random

import numpy as np

from surety.calib import metrics as m
from surety.eval.analyze import TAIL, _sets, _split, _table, variants
from surety.gate import AdaptiveRisk, FittedThreshold, PerChoiceThreshold, RiskConformal, SplitConformal

POLICIES = (FittedThreshold, PerChoiceThreshold, SplitConformal, RiskConformal)
TESTS = ("test_in", "test_shift")


def resplit(rows: list[dict], seed: int) -> list[dict]:
    pool = [i for i, r in enumerate(rows) if r["split"] in ("cal", "test_in")]
    cal = set(random.Random(seed).sample(pool, sum(r["split"] == "cal" for r in rows)))
    return [r if r["split"] == "test_shift" else {**r, "split": "cal" if i in cal else "test_in"}
            for i, r in enumerate(rows)]


def draws(rows: list[dict], eps_list=(0.05, 0.1), n_seeds: int = 100) -> list[dict]:
    out = []
    for seed in range(n_seeds):
        rs = resplit(rows, seed)
        probs = variants(rs)["debiased+T"]
        cal = _split(rs, probs, "cal")
        for eps in eps_list:
            for cls in POLICIES:
                policy = cls(eps)
                policy.fit(*cal)
                for split in TESTS:
                    p, g = _split(rs, probs, split)
                    sets = _sets(policy, p)
                    out.append({"seed": seed, "eps": eps, "policy": policy.name, "split": split,
                                "auto_err": m.autonomous_error_rate(sets, g), "automation": m.automation_rate(sets)})
    return out


def _pct(x) -> str:
    lo, hi = np.percentile(x, [2.5, 97.5])
    return f"[{lo:.3f}, {hi:.3f}]"


def draws_table(records: list[dict]) -> str:
    keys = sorted({(r["eps"], r["policy"], r["split"]) for r in records},
                  key=lambda k: (k[0], [c(k[0]).name for c in POLICIES].index(k[1]), k[2]))
    body = []
    for eps, policy, split in keys:
        rs = [r for r in records if (r["eps"], r["policy"], r["split"]) == (eps, policy, split)]
        err, auto = np.array([r["auto_err"] for r in rs]), np.array([r["automation"] for r in rs])
        body.append([str(eps), policy, split, len(rs), float(err.mean()), _pct(err), float(np.mean(err > eps)),
                     float(auto.mean()), _pct(auto)])
    return _table(("eps", "policy", "split", "draws", "auto_err", "auto_err_2.5–97.5%", "P(auto_err>eps)",
                   "automation", "automation_2.5–97.5%"), body)


def _site_ci(err: np.ndarray, act: np.ndarray, sites: list[str], n_boot: int, rng: np.random.Generator):
    names = sorted(set(sites))
    k = np.array([names.index(s) for s in sites])
    n_s, e_s, a_s = (np.bincount(k, weights=w, minlength=len(names)) for w in (np.ones(len(k)), err, act))
    pick = rng.integers(0, len(names), size=(n_boot, len(names)))
    n_b = n_s[pick].sum(1)
    return np.percentile(e_s[pick].sum(1) / n_b, [2.5, 97.5]), np.percentile(a_s[pick].sum(1) / n_b, [2.5, 97.5])


def site_bootstrap(rows: list[dict], eps_list=(0.05, 0.1), n_boot: int = 2000, seed: int = 0) -> str:
    probs = variants(rows)["debiased+T"]
    cal = _split(rows, probs, "cal")
    rng = np.random.default_rng(seed)
    body = []
    for eps in eps_list:
        for cls in POLICIES:
            policy = cls(eps)
            policy.fit(*cal)
            for split in TESTS:
                idx = [i for i, r in enumerate(rows) if r["split"] == split]
                sets = _sets(policy, [probs[i] for i in idx])
                gold = [rows[i]["gold_id"] for i in idx]
                err = np.array([len(s) == 1 and s[0] != g for s, g in zip(sets, gold)], float)
                act = np.array([len(s) == 1 for s in sets], float)
                sites = [rows[i]["site"] for i in idx]
                (elo, ehi), (alo, ahi) = _site_ci(err, act, sites, n_boot, rng)
                body.append([str(eps), policy.name, split, len(idx), len(set(sites)), float(err.mean()),
                             f"[{elo:.3f}, {ehi:.3f}]", float(act.mean()), f"[{alo:.3f}, {ahi:.3f}]"])
    return _table(("eps", "policy", "split", "n", "sites", "auto_err", "site_boot_95ci", "automation",
                   "site_boot_95ci"), body)


def _stream_order(rows: list[dict], seed: int) -> list[int]:
    sites = sorted({r["site"] for r in rows if r["split"] == "test_shift"})
    random.Random(seed).shuffle(sites)
    block = {s: k for k, s in enumerate(sites)}
    return sorted((i for i, r in enumerate(rows) if r["split"] == "test_shift"), key=lambda i: block[rows[i]["site"]])


def adaptive_draws(rows: list[dict], eps: float, audit_rates=(0.05, 0.2), gamma: float = 0.02,
                   n_seeds: int = 100) -> str:
    probs = variants(rows)["debiased+T"]
    cal = _split(rows, probs, "cal")
    body = []
    for r in audit_rates:
        err, tail, auto = [], [], []
        for seed in range(n_seeds):
            order = _stream_order(rows, seed)
            p, g = [probs[i] for i in order], [rows[i]["gold_id"] for i in order]
            policy = AdaptiveRisk(eps, gamma)
            policy.fit(*cal)
            rng = random.Random(seed)
            sets = []
            for x, y in zip(p, g):
                c = policy.prediction_set(x)
                sets.append(c)
                w = 1.0 if len(c) != 1 else (1 / r if rng.random() < r else 0.0)
                policy.update(x, y, w)
            err.append(m.autonomous_error_rate(sets, g))
            tail.append(m.autonomous_error_rate(sets[-TAIL:], g[-TAIL:]))
            auto.append(m.automation_rate(sets))
        err, tail, auto = map(np.array, (err, tail, auto))
        body.append([f"arisk@{eps}", f"{r:g}", n_seeds, float(err.mean()), _pct(err), float(np.mean(err > eps)),
                     float(tail.mean()), float(auto.mean()), _pct(auto)])
    return _table(("policy", "audit_rate", "streams", "auto_err", "auto_err_2.5–97.5%", "P(auto_err>eps)",
                   "tail_auto_err", "automation", "automation_2.5–97.5%"), body)
