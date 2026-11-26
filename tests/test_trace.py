import json

from surety.trace import Trace


def test_two_records_two_lines_with_ts(tmp_path):
    t = Trace(tmp_path / "runs" / "t.jsonl")
    t.record({"step": 0})
    t.record({"step": 1})
    t.close()
    rows = [json.loads(x) for x in (tmp_path / "runs" / "t.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == [0, 1] and all("ts" in r for r in rows)


def test_context_manager_writes_row_and_closes_file(tmp_path):
    path = tmp_path / "runs" / "t.jsonl"
    with Trace(path) as t:
        t.record({"x": 1})
    assert t.f.closed
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    assert [r["x"] for r in rows] == [1] and "ts" in rows[0]


def test_context_manager_closes_file_and_propagates_exception(tmp_path):
    path = tmp_path / "t.jsonl"
    try:
        with Trace(path) as t:
            t.record({"x": 1})
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    else:
        raise AssertionError("exception was swallowed")
    assert t.f.closed
    assert len(path.read_text().splitlines()) == 1
