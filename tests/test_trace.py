import json

from surety.trace import Trace


def test_two_records_two_lines_with_ts(tmp_path):
    t = Trace(tmp_path / "runs" / "t.jsonl")
    t.record({"step": 0})
    t.record({"step": 1})
    t.close()
    rows = [json.loads(x) for x in (tmp_path / "runs" / "t.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == [0, 1] and all("ts" in r for r in rows)
