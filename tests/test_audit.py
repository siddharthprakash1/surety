import random

import numpy as np
import pytest

from surety.eval.audits import audit_streams, pick, run_stream, site_stream
from surety.eval.resample import _stream_order
from surety.gate import AdaptiveRisk
from surety.gate.audit import AuditPolicy, _scale
from surety.gate.learned import AdaptiveLearnedRisk


def _probs(n, seed=0):
    """Steps with 3 options; the top option is right with probability p_top, so confident steps err less."""
    rng = random.Random(seed)
    probs, gold = [], []
    for _ in range(n):
        top = rng.uniform(0.34, 0.99)
        rest = (1 - top) * rng.uniform(0.5, 1.0)
        p = {"A": top, "B": rest, "C": 1 - top - rest}
        probs.append(p)
        gold.append("A" if rng.random() < top else rng.choice("BC"))
    return probs, gold


def _rows(n=600, n_sites=30, seed=0):
    probs, gold = _probs(n, seed)
    splits = ["cal", "test_in", "test_shift"]
    return [{"id": f"r{i}", "split": splits[i % 3], "site": f"s{(i // 3) % n_sites}.com", "gold_id": g, "probs": p,
             "task_type": "action_prediction"} for i, (p, g) in enumerate(zip(probs, gold))]


def _raw(rows):
    return [r["probs"] for r in rows]


def test_scale_hits_the_budget_and_its_edge_cases():
    v = np.array([0.0, 0.1, 0.2, 0.4, 0.8])
    s = _scale(v, 0.2, 0.05)
    assert float(np.clip(s * v, 0.05, 1).mean()) == pytest.approx(0.2)
    assert _scale(v, 0.05, 0.05) == 0.0                  # floor = budget: everyone at the floor
    assert _scale(np.zeros(4), 0.2, 0.05) is None and _scale(np.array([]), 0.2, 0.05) is None


def test_uniform_and_targeted_audit_probabilities():
    probs, gold = _probs(400)
    assert AuditPolicy(0.2).prob(probs[0]) == 0.2
    audit = AuditPolicy(0.2, targeted=True, floor=0.05)
    acted = [p for p in probs if max(p.values()) > 0.7]
    audit.fit(probs, gold, acted)
    q = np.array([audit.prob(p) for p in acted])
    assert q.mean() == pytest.approx(0.2) and q.min() >= 0.05 and q.max() <= 1
    e = np.array([audit.scorer.score(p) for p in acted])   # riskier steps are audited more (ê < 0.5 here)
    assert e.max() < 0.5 and np.all(np.diff(q[np.argsort(e)]) >= -1e-12)
    ipw = AuditPolicy(0.2, targeted=True, feedback="ipw")
    ipw.fit(probs, gold, acted)
    assert np.mean([ipw.prob(p) for p in acted]) == pytest.approx(0.2)


def test_targeted_falls_back_to_uniform_when_the_model_cannot_fit():
    probs, _ = _probs(50)
    audit = AuditPolicy(0.2, targeted=True)
    audit.fit(probs, [max(p, key=p.get) for p in probs], probs)   # never wrong: one class, no error model
    assert audit.scale is None and audit.prob(probs[0]) == 0.2
    with pytest.raises(ValueError):
        AuditPolicy(0.2, floor=0.3)
    with pytest.raises(ValueError):
        AuditPolicy(0.2, feedback="aipw")


def test_weighted_audits_are_unbiased():
    q, loss, rng = 0.07, 1.0, random.Random(0)
    est = [loss / q if rng.random() < q else 0.0 for _ in range(200_000)]
    assert np.mean(est) == pytest.approx(loss, rel=0.03)


def test_uniform_streams_match_learned_stream_draws_coin_for_coin():
    rows = _rows()
    cal = ([r["probs"] for r in rows if r["split"] == "cal"], [r["gold_id"] for r in rows if r["split"] == "cal"])
    for make in (lambda: AdaptiveRisk(0.1), lambda: AdaptiveLearnedRisk(0.1)):
        for seed in range(3):
            order = site_stream(rows, "test_shift", seed)
            assert order == _stream_order(rows, seed)
            p, g = [rows[i]["probs"] for i in order], [rows[i]["gold_id"] for i in order]
            old = make()
            old.fit(*cal)
            rng, sets = random.Random(seed), []               # learned.stream_draws' loop
            for x, y in zip(p, g):
                c = old.prediction_set(x)
                sets.append(c)
                old.update(x, y, 1.0 if len(c) != 1 else (1 / 0.2 if rng.random() < 0.2 else 0.0))
            new = make()
            new.fit(*cal)
            got = run_stream(new, AuditPolicy(0.2), p, g, seed)
            acted = [c for c in sets if len(c) == 1]
            assert got["automation"] == pytest.approx(len(acted) / len(sets))
            assert got["auto_err"] == pytest.approx(sum(len(c) == 1 and y not in c for c, y in zip(sets, g)) / len(g))


def test_audit_streams_share_streams_across_configs_and_pick_follows_the_rule():
    rows = _rows()
    configs = {"u": (lambda e, t, r: AdaptiveLearnedRisk(e), lambda r: AuditPolicy(r)),
               "t": (lambda e, t, r: AdaptiveLearnedRisk(e), lambda r: AuditPolicy(r, targeted=True))}
    held = audit_streams(rows, 0.1, configs, "held_out", 0.2, 3, probs_fn=_raw)
    cal = audit_streams(rows, 0.1, configs, "calibration", 0.2, 3, probs_fn=_raw)
    assert len(held) == len(cal) == 6 and {r["kind"] for r in cal} == {"calibration"}
    assert all(0 <= r["audit_rate"] <= 1 and 0 <= r["calls"] <= 1 for r in held + cal)
    recs = [{"config": n, "eps": 0.1, "auto_err": e, "automation": a} for n, e, a in
            [("ref", 0.05, 0.3), ("ref", 0.12, 0.3), ("x", 0.05, 0.5), ("x", 0.05, 0.4), ("y", 0.2, 0.9), ("y", 0.2, 0.9)]]
    assert pick(recs, "ref", ["x", "y"]) == "x" and pick(recs, "ref", ["y"]) is None
