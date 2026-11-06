from surety.eval.analyze import h3_table, select_escalated
from surety.gate import SplitConformal
from surety.supervise.client import MockSupervisor


def _row(i, split, tt, k, gold="o0"):
    p = {f"o{j}": 1 / k for j in range(k)}
    return {"id": f"r{i}", "split": split, "task_type": tt, "gold_id": gold, "option_ids": list(p),
            "probs": p, "runs": [p]}


def _res(cond, correct, pt, in_set=True, tt="action_prediction", chosen="o0"):
    return {"id": "x", "cond": cond, "task_type": tt, "chosen": chosen, "correct": correct,
            "gold_in_set": in_set, "prompt_tokens": pt, "completion_tokens": 10, "latency_ms": 100.0}


def test_h3_table_two_conditions():
    rows = [_res("full", True, 1000), _res("full", False, 1000, tt="element_grounding"),
            _res("set", True, 400), _res("set", False, 600, tt="element_grounding", chosen=None)]
    out = h3_table(rows)
    assert "| full | 2 | 0.500 | 0.000 | 0.500 |" in out
    assert "| set | 2 | 0.500 | 0.500 | 1.000 |" in out
    assert "500.000" in out and "gold_in_set" in out and "element_grounding n=1" in out


def test_offline_supervisor_selects_proper_subsets():
    rows = [_row(i, "test_in", "action_prediction" if i % 2 else "element_grounding", 4) for i in range(6)]
    rows.append(_row(9, "cal", "action_prediction", 4))
    rows.append(_row(10, "test_in", "action_prediction", 1))
    pol = SplitConformal(0.1)
    pol.fit([r["probs"] for r in rows if r["split"] == "cal"], ["o0"])
    pol.q = 0.8  # keeps all 4 uniform options -> full set, not a proper subset
    assert select_escalated(rows, pol, ["test_in"], 4, 0, [r["probs"] for r in rows]) == []
    for r in rows:
        r["probs"] = {"o0": 0.5, "o1": 0.3, "o2": 0.1, "o3": 0.1}
    pol.q = 0.8
    sel = select_escalated(rows, pol, ["test_in"], 4, 0, [r["probs"] for r in rows])
    assert len(sel) == 4 and all(2 <= len(c) < 4 and r["split"] == "test_in" for r, c in sel)
    assert {r["task_type"] for r, _ in sel} == {"action_prediction", "element_grounding"}
    assert MockSupervisor("o0").choose(None, sel[0][1])[0] == "o0"
