"""Uniform vs risk-targeted audits for the adaptive gates at the same audit budget (results/m1/*_audits.md).

Held-out streams are M1's: test_shift in blocks of sites, the gate fitted on M1's cal split (as learned.stream_draws).
Calibration-site streams, used to choose settings without looking at held-out sites: each seed re-draws cal from
cal ∪ test_in (resample.resplit), refits everything on it, and streams the remaining test_in steps in blocks of sites.
"""
import random

import numpy as np

from surety.calib import metrics as m
from surety.eval.analyze import TAIL, _split, _table
from surety.eval.learned import baseline
from surety.eval.resample import _pct, resplit
from surety.gate import AdaptiveRisk
from surety.gate.audit import AuditPolicy
from surety.gate.learned import AdaptiveLearnedRisk

KINDS = ("held_out", "calibration")


def site_stream(rows: list[dict], split: str, seed: int) -> list[int]:
    """The rows of `split` in blocks of sites, the site order shuffled by `seed` (resample._stream_order for test_shift)."""
    sites = sorted({r["site"] for r in rows if r["split"] == split})
    random.Random(seed).shuffle(sites)
    block = {s: k for k, s in enumerate(sites)}
    return sorted((i for i, r in enumerate(rows) if r["split"] == split), key=lambda i: block[rows[i]["site"]])


def run_stream(gate, audit: AuditPolicy, probs: list[dict], gold: list[str], seed: int) -> dict:
    """One stream. An escalated step is labelled (weight 1); an acted step is audited with probability q = audit.prob
    (weight 1/q) or not (weight 0). With a uniform policy the coins match learned.stream_draws draw for draw."""
    rng = random.Random(seed)
    sets, acted, audits = [], 0, 0
    for x, y in zip(probs, gold):
        c = gate.prediction_set(x)
        sets.append(c)
        w = 1.0
        if len(c) == 1:
            acted += 1
            q = audit.prob(x)
            w = 0.0
            if rng.random() < q:
                w, audits = 1 / q, audits + 1
        gate.update(x, y, w)
    t = len(gold)
    return {"auto_err": m.autonomous_error_rate(sets, gold), "automation": m.automation_rate(sets),
            "tail": m.autonomous_error_rate(sets[-TAIL:], gold[-TAIL:]),
            "audit_rate": audits / acted if acted else 0.0, "calls": (t - acted + audits) / t}


def audit_streams(rows: list[dict], eps: float, configs: dict, kind: str = "held_out", rate: float = 0.2,
                  n_seeds: int = 200, probs_fn=baseline) -> list[dict]:
    """configs: name -> (make_gate(eps, horizon, rate), make_audit(rate)); every config sees the same streams and
    audit coins. -> one record per (config, stream)."""
    if kind not in KINDS:
        raise ValueError(f"kind {kind!r} not in {KINDS}")
    if kind == "held_out":
        probs = probs_fn(rows)
        cal = _split(rows, probs, "cal")
    out = []
    for seed in range(n_seeds):
        rs = rows
        if kind == "calibration":
            rs = resplit(rows, seed)
            probs = probs_fn(rs)
            cal = _split(rs, probs, "cal")
        order = site_stream(rs, "test_shift" if kind == "held_out" else "test_in", seed)
        p, g = [probs[i] for i in order], [rs[i]["gold_id"] for i in order]
        for name, (make_gate, make_audit) in configs.items():
            gate = make_gate(eps, len(order), rate)
            gate.fit(*cal)
            audit = make_audit(rate)
            audit.fit(*cal, acted=[x for x in cal[0] if len(gate.prediction_set(x)) == 1])
            out.append({"config": name, "eps": eps, "kind": kind, "seed": seed, **run_stream(gate, audit, p, g, seed)})
    return out


def streams_table(records: list[dict]) -> str:
    body = []
    for name in dict.fromkeys(r["config"] for r in records):
        rs = [r for r in records if r["config"] == name]
        eps = rs[0]["eps"]
        err, auto = np.array([r["auto_err"] for r in rs]), np.array([r["automation"] for r in rs])
        body.append([name, len(rs), float(err.mean()), _pct(err), float(np.mean(err > eps)),
                     float(np.mean([r["tail"] for r in rs])), float(auto.mean()), _pct(auto),
                     float(np.mean([r["audit_rate"] for r in rs])), float(np.mean([r["calls"] for r in rs]))])
    return _table(("config", "streams", "auto_err", "auto_err_2.5–97.5%", "P(auto_err>eps)", "tail_auto_err",
                   "automation", "automation_2.5–97.5%", "audits/acted", "supervisor calls/step"), body)


def exceedance(records: list[dict], name: str) -> float:
    rs = [r for r in records if r["config"] == name]
    return float(np.mean([r["auto_err"] > r["eps"] for r in rs]))


def alearned(margin: float):
    return lambda eps, t, r: AdaptiveLearnedRisk(eps, margin=margin)


def arisk(eps, t, r):
    return AdaptiveRisk(eps)


def uniform(rate):
    return AuditPolicy(rate)


def targeted(floor: float = 0.05, feedback: str = "dr"):
    return lambda rate: AuditPolicy(rate, targeted=True, floor=floor, feedback=feedback)


def mean_of(records: list[dict], name: str, key: str) -> float:
    return float(np.mean([r[key] for r in records if r["config"] == name]))


def pick(records: list[dict], reference: str, candidates) -> str | None:
    """The candidate with the most automation among those whose share of streams over eps is no higher than the
    reference's; None if none qualifies."""
    ok = [n for n in candidates if exceedance(records, n) <= exceedance(records, reference)]
    return max(ok, key=lambda n: mean_of(records, n, "automation")) if ok else None
