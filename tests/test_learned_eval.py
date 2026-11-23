import math
import random

import numpy as np
import pytest

from surety.eval.learned import ADAPTIVE, auroc, draws_table, ranking_table, static_draws, stream_draws


def _rows(n=300, seed=0):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        k = 3 if i % 2 else 6
        opts = [f"o{j}" for j in range(k)]
        gold = rng.choice(opts)
        sure = rng.random()
        top = gold if rng.random() < 0.3 + 0.6 * sure else rng.choice(opts)
        w = {o: (1 + 8 * sure if o == top else rng.random()) for o in opts}
        z = sum(w.values())
        probs = {o: v / z for o, v in w.items()}
        split = "cal" if i < 120 else "test_in" if i < 200 else "test_shift"
        rows.append({"split": split, "site": f"s{i % 10}", "task_type": "action_prediction" if i % 2 else "element_grounding",
                     "gold_id": gold, "probs": probs, "runs": [probs]})
    return rows


def test_auroc():
    wrong = np.array([0, 0, 1, 1], float)
    assert auroc(np.array([0.1, 0.2, 0.3, 0.4]), wrong) == 1.0
    assert auroc(np.array([0.4, 0.3, 0.2, 0.1]), wrong) == 0.0
    assert auroc(np.array([0.5, 0.5, 0.5, 0.5]), wrong) == 0.5
    assert auroc(np.array([0.1, 0.3, 0.2, 0.4]), wrong) == 0.75
    assert math.isnan(auroc(np.array([0.1, 0.2]), np.array([0.0, 0.0])))


def test_ranking_table_lists_both_test_splits_by_task_type():
    t = ranking_table(_rows())
    lines = [x for x in t.splitlines() if x.startswith("| test_")]
    assert len(lines) == 6 and "AUROC learned" in t


def test_static_draws_cover_every_policy_and_split():
    rec = static_draws(_rows(), eps_list=(0.1,), n_seeds=2)
    assert len(rec) == 2 * 3 * 2
    assert {r["policy"] for r in rec} == {"risk@0.1", "learned@0.1", "learned_pac@0.1"}
    assert all(0 <= r["auto_err"] <= r["automation"] <= 1 for r in rec)
    table = draws_table(rec)
    assert table.count("| 0.1 |") == 6


@pytest.mark.parametrize("eps", [0.1])
def test_stream_draws_one_row_per_gate(eps):
    t = stream_draws(_rows(), eps, ADAPTIVE, audit_rate=0.5, n_seeds=2)
    assert [x.split("|")[1].strip() for x in t.splitlines()[2:]] == list(ADAPTIVE)
