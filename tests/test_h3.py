import importlib.util
from pathlib import Path

from surety.eval.analyze import h3_table, select_escalated
from surety.gate import SplitConformal
from surety.supervise import MockSupervisor
from surety.types import Instance


def _row(i, split, tt):
    p = {"o0": 0.5, "o1": 0.3, "o2": 0.1, "o3": 0.1}
    return {"id": f"r{i}", "split": split, "task_type": tt, "gold_id": "o0", "option_ids": list(p),
            "probs": p, "runs": [p]}


def _res(cond, correct, in_set=True, tt="action_prediction", chosen="o0", **kw):
    return {"id": "x", "cond": cond, "task_type": tt, "chosen": chosen, "correct": correct,
            "gold_in_set": in_set, "prompt_tokens": 1000 if cond == "full" else 400, "completion_tokens": 10,
            "latency_ms": 100.0, **kw}


def _cols(out):
    lines = out.splitlines()
    head = [c.strip() for c in lines[0].strip("|").split("|")]
    return {r[0]: dict(zip(head, r)) for r in ([c.strip() for c in ln.strip("|").split("|")] for ln in lines[2:4])}


def test_h3_table_two_conditions():
    rows = [_res("full", True), _res("full", False, tt="element_grounding"),
            _res("set", True), _res("set", False, in_set=False, chosen=None, abstained=True),
            _res("set", False, in_set=False, chosen=None, abstained=False),
            _res("set", True, fallback_parsed=True)]
    out = h3_table(rows)
    t = _cols(out)
    assert t["full"]["n"] == "2" and t["full"]["acc"] == "0.500" and t["full"]["abstain"] == "0.000"
    assert t["set"]["n"] == "4" and t["set"]["acc"] == "0.500"
    assert t["set"]["abstain"] == "0.250" and t["set"]["unparsed"] == "0.250" and t["set"]["fallback"] == "0.250"
    assert t["set"]["acc_given_answer"] == "1.000" and t["set"]["acc_gold_in_set"] == "1.000"
    assert t["full"]["prompt_tok"] == "1000.000" and t["set"]["prompt_tok"] == "400.000"
    assert "0.500; action_prediction n=4" in out


def test_h3_table_legacy_null_counts_as_abstain():
    t = _cols(h3_table([_res("full", False, chosen=None), _res("set", True)]))
    assert t["full"]["abstain"] == "1.000" and t["full"]["unparsed"] == "0.000"


def test_offline_supervisor_selects_proper_subsets():
    rows = [_row(i, "test_in", "action_prediction" if i % 2 else "element_grounding") for i in range(8)]
    rows += [_row(20 + i, "test_shift", "action_prediction") for i in range(4)]
    rows += [_row(9, "cal", "action_prediction"), {**_row(11, "cal", "action_prediction"), "gold_id": "o1"}]
    full = _row(10, "test_in", "action_prediction")
    full["probs"] = {"o0": 0.5, "o1": 0.3, "o2": 0.3, "o3": 0.3}  # all four within q
    rows.append(full)
    cal = [r for r in rows if r["split"] == "cal"]
    pol = SplitConformal(0.5)
    pol.fit([r["probs"] for r in cal], [r["gold_id"] for r in cal])
    probs = [r["probs"] for r in rows]
    sel = select_escalated(rows, pol, ["test_in", "test_shift"], 6, 0, probs)
    assert len(sel) == 6 and all(2 <= len(c) < 4 for _, c in sel)
    assert full["id"] not in {r["id"] for r, _ in sel}
    count = lambda k: sum((r["split"], r["task_type"]) == k for r, _ in sel)  # noqa: E731
    assert count(("test_in", "action_prediction")) == 2 and count(("test_shift", "action_prediction")) == 2
    assert count(("test_in", "element_grounding")) == 2
    assert [r["id"] for r, _ in select_escalated(rows, pol, ["test_in", "test_shift"], 6, 0, probs)] == \
        [r["id"] for r, _ in sel]


def test_offline_supervisor_lists_set_in_page_order(tmp_path):
    spec = importlib.util.spec_from_file_location("h3", Path(__file__).parents[1] / "scripts/offline_supervisor.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    inst = Instance("x", "s", "action_prediction", "operation", ["A: a", "B: b", "C: c"], 0, "ctx", "q", 0, "text")
    row = {"id": "x", "split": "test_in", "task_type": "action_prediction", "option_ids": ["A", "B", "C"],
           "gold_id": "A"}
    sup = MockSupervisor("C")
    script.run([(row, ["C", "A"])], {"x": inst}, sup, tmp_path / "h3.jsonl")
    assert sup.candidates == [None, ["A", "C"]]
