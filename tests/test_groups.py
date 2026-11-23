import random

from surety.eval.groups import adaptive_group_streams, group_draws, group_table


def _rows(seed=0):
    """Two task types: 'easy' is confident and right, 'hard' is confident and often wrong."""
    rng = random.Random(seed)
    rows = []
    for i in range(400):
        split = ("cal", "test_in", "test_shift", "cal")[i % 4]
        kind = "easy" if (i // 4) % 2 else "hard"
        gold = rng.choice("abc")
        top = gold if rng.random() < (0.95 if kind == "easy" else 0.5) else rng.choice([o for o in "abc" if o != gold])
        w = {o: (8.0 if o == top else rng.random()) for o in "abc"}
        z = sum(w.values())
        p = {o: v / z for o, v in w.items()}
        rows.append({"split": split, "task_type": kind, "site": f"s{i % 13}", "gold_id": gold, "probs": p, "runs": [p]})
    return rows


def test_per_group_gate_controls_the_hard_group():
    recs = group_draws(_rows(), eps_list=(0.1,), n_seeds=5)
    assert {(r["gate"], r["group"]) for r in recs} == {(g, k) for g in ("marginal", "per_group")
                                                       for k in ("easy", "hard", "all")}
    mean = lambda gate, group: sum(r["auto_err"] for r in recs if (r["gate"], r["group"]) == (gate, group)) / 5  # noqa: E731
    assert mean("per_group", "hard") <= mean("marginal", "hard")
    table = group_table(recs)
    assert table.splitlines()[2].startswith("| 0.1 | marginal | all | 5 |")


def test_adaptive_group_streams_render():
    t = adaptive_group_streams(_rows(), 0.1, n_seeds=3)
    lines = t.splitlines()
    assert len(lines) == 2 + 6 and lines[2].startswith("| arisk@0.1 | one gate | all | 3 |")
