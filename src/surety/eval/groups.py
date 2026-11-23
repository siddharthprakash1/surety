"""Marginal vs group-conditional (per task type) risk control over calibration draws.

CRC bounds the autonomous error averaged over all steps; a type with more hard steps can exceed ε while another sits
far below. Fitting one RiskConformal per group on that group's cal rows bounds each group (Mondrian-style), at the
price of fewer calibration rows per fit. T stays pooled, as in M1.
"""
import random

import numpy as np

from surety.calib import metrics as m
from surety.eval.analyze import _sets, _table, variants
from surety.eval.resample import _stream_order, resplit
from surety.gate import AdaptiveRisk, RiskConformal


def group_draws(rows: list[dict], eps_list=(0.05, 0.1), n_seeds: int = 200, key: str = "task_type",
                split: str = "test_shift") -> list[dict]:
    groups = sorted({r[key] for r in rows})
    out = []
    for seed in range(n_seeds):
        rs = resplit(rows, seed)
        probs = variants(rs)["debiased+T"]
        cal = {g: [i for i, r in enumerate(rs) if r["split"] == "cal" and r[key] == g] for g in groups}
        test = {g: [i for i, r in enumerate(rs) if r["split"] == split and r[key] == g] for g in groups}
        all_cal = [i for g in groups for i in cal[g]]
        for eps in eps_list:
            marginal = RiskConformal(eps)
            marginal.fit([probs[i] for i in all_cal], [rs[i]["gold_id"] for i in all_cal])
            per = {}
            for g in groups:
                per[g] = RiskConformal(eps)
                per[g].fit([probs[i] for i in cal[g]], [rs[i]["gold_id"] for i in cal[g]])
            for name, gate_of in (("marginal", lambda g: marginal), ("per_group", lambda g: per[g])):
                sets, gold = {}, {}
                for g in groups:
                    sets[g] = _sets(gate_of(g), [probs[i] for i in test[g]])
                    gold[g] = [rs[i]["gold_id"] for i in test[g]]
                    out.append({"seed": seed, "eps": eps, "gate": name, "group": g,
                                "auto_err": m.autonomous_error_rate(sets[g], gold[g]),
                                "automation": m.automation_rate(sets[g])})
                both_s, both_g = sum(sets.values(), []), sum(gold.values(), [])
                out.append({"seed": seed, "eps": eps, "gate": name, "group": "all",
                            "auto_err": m.autonomous_error_rate(both_s, both_g),
                            "automation": m.automation_rate(both_s)})
    return out


def group_table(records: list[dict]) -> str:
    body = []
    for eps, gate, group in sorted({(r["eps"], r["gate"], r["group"]) for r in records}):
        rs = [r for r in records if (r["eps"], r["gate"], r["group"]) == (eps, gate, group)]
        err, auto = np.array([r["auto_err"] for r in rs]), np.array([r["automation"] for r in rs])
        lo, hi = np.percentile(err, [2.5, 97.5])
        body.append([str(eps), gate, group, len(rs), float(err.mean()), f"[{lo:.3f}, {hi:.3f}]",
                     float(np.mean(err > eps)), float(auto.mean())])
    return _table(("eps", "gate", "group", "draws", "auto_err", "auto_err_2.5–97.5%", "P(auto_err>eps)",
                   "automation"), body)


def adaptive_group_streams(rows: list[dict], eps: float, audit_rate: float = 0.2, gamma: float = 0.02,
                           n_seeds: int = 200, key: str = "task_type") -> str:
    """test_shift streams (site order and audits redrawn per stream): one AdaptiveRisk for all steps vs one per group."""
    probs = variants(rows)["debiased+T"]
    groups = sorted({r[key] for r in rows})
    cal = {g: [i for i, r in enumerate(rows) if r["split"] == "cal" and r[key] == g] for g in groups}
    every = [i for g in groups for i in cal[g]]
    res: dict[tuple, list] = {}
    for seed in range(n_seeds):
        order = _stream_order(rows, seed)
        for mode in ("one gate", "per group"):
            fits = {g: cal[g] for g in groups} if mode == "per group" else {"*": every}
            gates = {}
            for name, idx in fits.items():
                gates[name] = AdaptiveRisk(eps, gamma)
                gates[name].fit([probs[i] for i in idx], [rows[i]["gold_id"] for i in idx])
            rng, sets, gold = random.Random(seed), {g: [] for g in groups}, {g: [] for g in groups}
            for i in order:
                g, x, y = rows[i][key], probs[i], rows[i]["gold_id"]
                gate = gates[g if mode == "per group" else "*"]
                c = gate.prediction_set(x)
                sets[g].append(c)
                gold[g].append(y)
                gate.update(x, y, 1.0 if len(c) != 1 else (1 / audit_rate if rng.random() < audit_rate else 0.0))
            for g in [*groups, "all"]:
                s, y = (sum(sets.values(), []), sum(gold.values(), [])) if g == "all" else (sets[g], gold[g])
                res.setdefault((mode, g), []).append((m.autonomous_error_rate(s, y), m.automation_rate(s)))
    body = []
    for (mode, g), v in sorted(res.items()):
        err, auto = np.array([x[0] for x in v]), np.array([x[1] for x in v])
        body.append([f"arisk@{eps}", mode, g, len(v), float(err.mean()), float(np.mean(err > eps)), float(auto.mean())])
    return _table(("policy", "gate", "group", "streams", "auto_err", "P(auto_err>eps)", "automation"), body)
