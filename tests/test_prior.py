import math

import numpy as np
import pytest

from surety.decide import PriorCorrector


def biased(n, seed=0, bias=math.e ** 2):
    """Gold uniform over 4 actions; the executor sees it (0.6 on gold) but inflates STOP by `bias`."""
    rng = np.random.default_rng(seed)
    acts = ["CLICK", "TYPE", "SCROLL", "STOP"]
    probs, gold = [], []
    for _ in range(n):
        g = acts[rng.integers(4)]
        w = {a: (0.6 if a == g else 0.4 / 3) * (bias if a == "STOP" else 1.0) * rng.uniform(0.7, 1.3) for a in acts}
        z = sum(w.values())
        probs.append({a: v / z for a, v in w.items()})
        gold.append(g)
    return probs, gold


def accuracy(probs, gold, f=lambda p: p):
    return np.mean([max(q := f(p), key=q.get) == g for p, g in zip(probs, gold)])


def test_prior_correction_removes_a_label_bias():
    (cal_p, cal_g), (test_p, test_g) = biased(600), biased(600, seed=1)
    pc = PriorCorrector(l2=0.1)
    b = pc.fit(cal_p, cal_g)
    # only relative offsets matter: STOP sits ~2 below the others
    others = np.mean([b[a] for a in ("CLICK", "TYPE", "SCROLL")])
    assert b["STOP"] - others == pytest.approx(-2.0, abs=0.35)
    assert accuracy(test_p, test_g) < 0.6 < accuracy(test_p, test_g, pc.apply)


def test_element_ids_and_rare_ids_keep_no_offset():
    probs = [{"E1": 0.5, "e2": 0.2, "bid:7": 0.1, "clickable_3": 0.1, "DONE": 0.1}] * 40 + [{"RARE": 0.5, "DONE": 0.5}] * 5
    gold = ["E1"] * 40 + ["RARE"] * 5
    pc = PriorCorrector(min_rows=30)
    assert set(pc.fit(probs, gold)) == {"DONE"}
    out = pc.apply({"E9": 0.5, "DONE": 0.5})
    assert sum(out.values()) == pytest.approx(1.0) and set(out) == {"E9", "DONE"}


def test_unfitted_or_empty_vocabulary_is_the_identity():
    pc = PriorCorrector()
    p = {"a": 0.7, "b": 0.3}
    assert pc.apply(p) == p and pc.apply(p) is not p
    assert pc.fit([{"E1": 0.6, "E2": 0.4}] * 50, ["E1"] * 50) == {} and pc.apply(p) == p


def test_fit_is_stable_with_extreme_probabilities():
    # near-zero probabilities on gold made undamped Newton overshoot (offsets of ±100); the line search keeps it sane
    probs = [{"CLICK": 1e-12, "DONE": 1 - 1e-12}] * 50 + [{"CLICK": 0.9, "DONE": 0.1}] * 50
    gold = ["CLICK"] * 100
    b = PriorCorrector(l2=1.0).fit(probs, gold)
    assert all(abs(v) < 30 for v in b.values()) and b["CLICK"] > b["DONE"]


def _rows(n=240, seed=0):
    """WebJev-like rows: half action prediction with a BLOCKED-inflating executor, half grounding."""
    rng = np.random.default_rng(seed)
    acts = ["CLICK", "TYPE_TEXT", "SCROLL_DOWN", "BLOCKED"]
    rows = []
    for i in range(n):
        split = "cal" if i < n // 2 else "test_in" if i < 3 * n // 4 else "test_shift"
        if i % 2:
            g = acts[rng.integers(3)]  # BLOCKED is never gold
            w = {a: (0.5 if a == g else 0.15) * (6.0 if a == "BLOCKED" else 1.0) for a in acts}
            task = "action_prediction"
        else:
            g = f"E{rng.integers(1, 4)}"
            w = {f"E{k}": (0.6 if f"E{k}" == g else 0.2) for k in range(1, 4)}
            task = "element_grounding"
        z = sum(w.values())
        p = {o: v / z for o, v in w.items()}
        rows.append({"split": split, "site": f"s{i % 7}", "task_type": task, "gold_id": g, "probs": p, "runs": [p]})
    return rows


def test_prior_pipeline_fixes_the_synthetic_bias_and_feeds_the_draws():
    from surety.eval.learned import draws_table, static_draws
    from surety.eval.prior import accuracy_table, prior

    rows = _rows()
    probs = prior(rows)
    assert all(abs(sum(p.values()) - 1) < 1e-9 for p in probs)
    ap = [i for i, r in enumerate(rows) if r["task_type"] == "action_prediction" and r["split"] != "cal"]
    before = np.mean([max(rows[i]["probs"], key=rows[i]["probs"].get) == rows[i]["gold_id"] for i in ap])
    after = np.mean([max(probs[i], key=probs[i].get) == rows[i]["gold_id"] for i in ap])
    assert before < 0.2 and after > 0.9
    table = accuracy_table(rows)
    assert table.count("| test_shift |") == 3 and "acc prior+T" in table
    rec = static_draws(rows, eps_list=(0.1,), n_seeds=2, probs_fn=prior, label=" +prior")
    assert {r["policy"] for r in rec} == {"risk@0.1 +prior", "learned@0.1 +prior", "learned_pac@0.1 +prior"}
    assert draws_table(rec).count("+prior") == 6
