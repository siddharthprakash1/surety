import importlib.util
import io
import json
from pathlib import Path

from surety.eval.analyze import h3_table, select_escalated
from surety.gate import SplitConformal
from surety.supervise import MockSupervisor, RateLimited
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


def test_h3_table_paired_conditions():
    rows = [_res("full", True, id="a"), _res("full", False, tt="element_grounding", id="b"),
            _res("full", True, id="c"), _res("full", True, id="d"), _res("full", False, id="only_full"),
            _res("set", True, id="a"), _res("set", False, in_set=False, chosen=None, abstained=True, id="b"),
            _res("set", False, in_set=False, chosen=None, abstained=False, id="c"),
            _res("set", True, fallback_parsed=True, id="d"), _res("set", False, id="d"),
            _res("set", True, id="only_set")]
    out = h3_table(rows)
    t = _cols(out)
    assert t["full"]["n"] == "4" and t["full"]["acc"] == "0.750" and t["full"]["abstain"] == "0.000"
    assert t["set"]["n"] == "4" and t["set"]["acc"] == "0.500"
    assert t["set"]["abstain"] == "0.250" and t["set"]["unparsed"] == "0.250" and t["set"]["fallback"] == "0.250"
    assert t["set"]["acc_given_answer"] == "1.000" and t["set"]["acc_gold_in_set"] == "1.000"
    assert t["full"]["prompt_tok"] == "1000" and t["set"]["prompt_tok"] == "400"
    assert "H3 sample): 0.500; paired n=4; action_prediction n=4" in out
    assert ("both answered n=2: acc full 1.000 / set 1.000; discordant full-only 0 / set-only 0; "
            "exact McNemar p=1.000") in out


def test_mcnemar_exact():
    from surety.eval.analyze import _mcnemar
    assert _mcnemar(5, 1) == 14 / 64 and _mcnemar(0, 0) == 1.0 and _mcnemar(3, 3) == 1.0


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


def _script():
    spec = importlib.util.spec_from_file_location("h3", Path(__file__).parents[1] / "scripts/offline_supervisor.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


_INST = Instance("x", "s", "action_prediction", "operation", ["A: a", "B: b", "C: c"], 0, "ctx", "q", 0, "text")
_ROW = {"id": "x", "split": "test_in", "task_type": "action_prediction", "option_ids": ["A", "B", "C"], "gold_id": "A"}


def test_offline_supervisor_lists_set_in_page_order(tmp_path):
    sup = MockSupervisor("C")
    _script().run([(_ROW, ["C", "A"])], {"x": _INST}, sup, tmp_path / "h3.jsonl")
    assert sup.candidates == [None, ["A", "C"]]


class _CappedAfter(MockSupervisor):
    def __init__(self, n):
        super().__init__("C")
        self.n = n

    def choose(self, state, candidate_ids=None):
        if self.calls == self.n:
            raise RateLimited(3600.0)
        return super().choose(state, candidate_ids)


def test_offline_supervisor_stops_at_a_daily_cap_and_resumes(tmp_path):
    script, out, log = _script(), tmp_path / "h3.jsonl", io.StringIO()
    assert script.run([(_ROW, ["C", "A"])], {"x": _INST}, _CappedAfter(1), out, log=log) is False
    assert len(out.read_text().splitlines()) == 1 and "stopped: rate limited" in log.getvalue()
    assert script.run([(_ROW, ["C", "A"])], {"x": _INST}, MockSupervisor("C"), out, log=log) is True
    assert [json.loads(x)["cond"] for x in out.read_text().splitlines()] == ["full", "set"]
    assert log.getvalue().endswith("2/2 calls saved\n")
