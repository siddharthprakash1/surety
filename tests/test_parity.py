import json

import pytest

from surety.eval.parity import load_rows, parity


def _row(id, probs, gold, options=None):
    return {"id": id, "probs": probs, "gold_id": gold, "option_ids": options or list(probs)}


def test_parity_compares_shared_steps(tmp_path):
    a = {"x": _row("x", {"A": 0.7, "B": 0.3}, "A"), "y": _row("y", {"A": 0.4, "B": 0.6}, "A"),
         "z": _row("z", {"A": 1.0}, "A")}
    b = {"x": _row("x", {"A": 0.6, "B": 0.4}, "A"), "y": _row("y", {"A": 0.6, "B": 0.4}, "A", ["B", "A"]),
         "w": _row("w", {"A": 1.0}, "A")}
    p = parity(a, b)
    assert (p["shared"], p["only_a"], p["only_b"]) == (2, 1, 1)
    assert (p["top1_agree"], p["acc_a"], p["acc_b"], p["same_options"]) == (0.5, 0.5, 1.0, 0.5)
    assert p["mean_abs_diff"] == pytest.approx(0.15) and p["max_abs_diff"] == pytest.approx(0.2)
    assert parity(a, {}) == {"shared": 0, "only_a": 3, "only_b": 0}
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in a.values()) + "\n", encoding="utf-8")
    assert load_rows(path) == a
