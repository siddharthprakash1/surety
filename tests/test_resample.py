import random

import numpy as np

from surety.eval.resample import _site_ci, adaptive_draws, draws, draws_table, resplit, site_bootstrap


def _rows(n_cal=60, n_in=40, n_shift=60, k=4, seed=0):
    rng = random.Random(seed)
    rows = []
    for i, split in enumerate(["cal"] * n_cal + ["test_in"] * n_in + ["test_shift"] * n_shift):
        ids = [f"o{j}" for j in range(k)]
        gold = rng.choice(ids)
        top = gold if rng.random() < 0.6 else rng.choice(ids)
        w = {o: (6.0 if o == top else rng.random()) for o in ids}
        z = sum(w.values())
        p = {o: v / z for o, v in w.items()}
        rows.append({"id": str(i), "split": split, "site": f"{split}-{i % 7}", "gold_id": gold, "option_ids": ids,
                     "probs": p, "runs": [p]})
    return rows


def test_resplit_keeps_sizes_and_shift_rows():
    rows = _rows()
    a, b = resplit(rows, 1), resplit(rows, 2)
    assert [sum(r["split"] == s for r in a) for s in ("cal", "test_in", "test_shift")] == [60, 40, 60]
    assert [r for r in a if r["split"] == "test_shift"] == [r for r in rows if r["split"] == "test_shift"]
    assert a != b and resplit(rows, 1) == a
    assert {r["split"] for r in rows} == {"cal", "test_in", "test_shift"}  # input untouched


def test_draws_records_and_table():
    recs = draws(_rows(), eps_list=(0.1,), n_seeds=3)
    assert len(recs) == 3 * 4 * 2
    assert {r["policy"] for r in recs} == {"fitted@0.1", "per_choice@0.1", "conformal@0.1", "risk@0.1"}
    table = draws_table(recs)
    assert table.splitlines()[2].startswith("| 0.1 | fitted@0.1 | test_in | 3 |")
    assert len(table.splitlines()) == 2 + 8


def test_site_ci_is_ratio_of_resampled_site_sums():
    err, act = np.array([1, 0, 0, 0], float), np.array([1, 1, 0, 1], float)
    (elo, ehi), (alo, ahi) = _site_ci(err, act, ["a", "a", "b", "b"], 4000, np.random.default_rng(0))
    # sites a: 1/2 err, 2/2 act; b: 0/2, 1/2 -> resampled ratios in {0, 0.25, 0.5} and {0.5, 0.75, 1}
    assert (elo, ehi) == (0.0, 0.5) and (alo, ahi) == (0.5, 1.0)


def test_site_bootstrap_and_adaptive_tables_render():
    rows = _rows()
    sb = site_bootstrap(rows, eps_list=(0.1,), n_boot=50)
    assert len(sb.splitlines()) == 2 + 8 and "| 0.1 | risk@0.1 | test_shift | 60 | 7 |" in sb
    ad = adaptive_draws(rows, 0.1, audit_rates=(0.2,), n_seeds=4)
    assert ad.splitlines()[2].startswith("| arisk@0.1 | 0.2 | 4 |")
