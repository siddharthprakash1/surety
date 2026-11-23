import random

from surety.eval.sweep import eps_sweep, sweep_plot, sweep_table


def _rows(n=160, seed=0):
    rng = random.Random(seed)
    rows = []
    for i, split in enumerate(["cal"] * (n // 2) + ["test_in"] * (n // 4) + ["test_shift"] * (n // 4)):
        gold = rng.choice("abcd")
        top = gold if rng.random() < 0.6 else rng.choice("abcd")
        w = {o: (6.0 if o == top else rng.random()) for o in "abcd"}
        z = sum(w.values())
        p = {o: v / z for o, v in w.items()}
        rows.append({"split": split, "site": f"s{i % 9}", "gold_id": gold, "probs": p, "runs": [p]})
    return rows


def test_sweep_records_table_and_plot(tmp_path):
    recs = eps_sweep(_rows(), eps_grid=(0.05, 0.1), n_seeds=2)
    assert len(recs) == 2 * 2 * 3 and {r["policy"] for r in recs} == {"fitted", "per_choice", "risk"}
    table = sweep_table(recs)
    assert table.splitlines()[2].startswith("| 0.05 | fitted |") and len(table.splitlines()) == 2 + 6
    sweep_plot(recs, tmp_path / "s.png")
    assert (tmp_path / "s.png").stat().st_size > 0
