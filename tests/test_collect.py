import json

import httpx
import pytest

from surety.decide import MockBackend
from surety.eval.collect import CollectAborted, collect
from surety.types import Instance


def inst(i):
    return Instance(id=f"i{i}", site="s", task_type="action_prediction", question_type="operation",
                    options=["0: click", "1: type"], gold=1, context="ctx", question="q",
                    num_tokens=1, context_format="text")


def rows(p):
    return [json.loads(line) for line in p.read_text().splitlines()]


def test_collect_writes_rows(tmp_path):
    out = tmp_path / "o.jsonl"
    n = collect([inst(0), inst(1)], MockBackend(), "full", 3, out, {"i0": "cal", "i1": "test_in"})
    r = rows(out)
    assert n == 2 and [x["split"] for x in r] == ["cal", "test_in"]
    assert r[0]["gold_id"] == "1" and set(r[0]["probs"]) == set(r[0]["option_ids"])
    assert len(r[0]["runs"]) == 3 and r[0]["n_options"] == 2 and r[0]["backend"] == "mock"


def test_collect_resumes(tmp_path):
    out, split = tmp_path / "o.jsonl", {"i0": "cal", "i1": "cal", "i2": "cal"}
    assert collect([inst(0), inst(1)], MockBackend(), "full", 1, out, split) == 2
    b = MockBackend()
    assert collect([inst(0), inst(1)], b, "full", 1, out, split) == 0 and not b.calls
    assert collect([inst(0), inst(1), inst(2)], b, "full", 1, out, split) == 1
    assert len(rows(out)) == 3


def test_collect_no_resume_overwrites(tmp_path):
    out, split = tmp_path / "o.jsonl", {"i0": "cal"}
    collect([inst(0)], MockBackend(), "full", 1, out, split)
    assert collect([inst(0)], MockBackend(), "full", 1, out, split, resume=False) == 1
    assert len(rows(out)) == 1


def test_collect_skips_backend_errors(tmp_path):
    def fn(state):
        if state.meta["gold_id"] == "1" and state.state_text.startswith("bad"):
            raise httpx.ConnectError("down")
        return {"0": 0.5, "1": 0.5}

    bad = inst(0)
    bad.context = "bad"
    out = tmp_path / "o.jsonl"
    n = collect([bad, inst(1)], MockBackend(fn), "full", 1, out, {"i0": "cal", "i1": "cal"})
    assert n == 1 and rows(out)[0]["id"] == "i1"


def test_collect_resume_repairs_torn_line(tmp_path):
    out, split = tmp_path / "o.jsonl", {"i0": "cal", "i1": "cal"}
    collect([inst(0)], MockBackend(), "full", 1, out, split)
    with out.open("a") as f:
        f.write('{"id": "i1", "mo')
    assert collect([inst(0), inst(1)], MockBackend(), "full", 1, out, split) == 1
    assert [x["id"] for x in rows(out)] == ["i0", "i1"]


def test_collect_resume_keys_on_mode(tmp_path):
    out, split = tmp_path / "o.jsonl", {"i0": "cal"}
    collect([inst(0)], MockBackend(), "full", 1, out, split)
    assert collect([inst(0)], MockBackend(), "compact", 1, out, split) == 1


def test_collect_aborts_on_consecutive_failures(tmp_path):
    def fn(state):
        raise httpx.ConnectError("down")

    insts = [inst(i) for i in range(30)]
    b = MockBackend(fn)
    with pytest.raises(CollectAborted):
        collect(insts, b, "full", 1, tmp_path / "o.jsonl", {i.id: "cal" for i in insts})
    assert len(b.calls) == 20


def test_collect_uses_fewer_perms_for_large(tmp_path):
    big = inst(0)
    big.options = [f"{i}: x" for i in range(5)]
    out = tmp_path / "o.jsonl"
    collect([big, inst(1)], MockBackend(), "full", 5, out, {"i0": "cal", "i1": "cal"}, n_perm_large=2, large_options=3)
    assert [x["n_perm"] for x in rows(out)] == [2, 5]
