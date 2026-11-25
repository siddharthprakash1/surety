import json
import socket
import subprocess
import sys
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from surety.eval.serve_collect import read_files, score_plan, serve_and_collect

ROOT = Path(__file__).resolve().parents[1]
AP, EG = "action_prediction", "element_grounding"
# a stand-in for kev.serve: /v1/models names the run, /v1/systemone(/permute) answer uniform probabilities
FAKE = r'''
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
port, run, mode = int(sys.argv[1]), sys.argv[2], sys.argv[3]
if mode == "die":
    sys.exit(3)

class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if mode == "busy":
            return self.send(503, {})
        self.send(200, {"models": [{"name": "kev-latest", "run": run}, {"name": "jev-latest", "run": run}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/v1/systemone/permute":
            ids = list(body["request"]["questions"]["q"]["criteria"])
            runs = [{"probabilities": {i: 1 / len(ids) for i in ids}} for _ in range(body["n_perm"])]
            return self.send(200, {"runs": runs, "usage": {"input_tokens": 7}})
        ids = list(body["questions"]["q"]["criteria"])
        self.send(200, {"answers": {"q": {"probabilities": {i: 1 / len(ids) for i in ids}}}, "usage": {"input_tokens": 7}})

HTTPServer(("127.0.0.1", port), H).serve_forever()
'''


def _port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _server(tmp_path, port, run, mode="serve"):
    script = tmp_path / "fake_serve.py"
    script.write_text(FAKE)
    return [sys.executable, str(script), str(port), run, mode]


def _webjev(tmp_path) -> Path:
    elements = [{"role": "button", "value": "", "index": str(k), "label": f"Button {k}", "operations": ["CLICK"]}
                for k in range(1, 6)]
    context = json.dumps({"page": {"url": "https://shop.test/socks", "title": "Socks", "text": "sock " * 50},
                          "elements": elements, "recent_actions": []})
    question = json.dumps({"goal": "The user's original task (global constraints; no stage may drop any of its "
                                   "conditions): Buy socks.", "operation": "CLICK"})
    common = {"context": context, "question": question, "num_tokens": 0, "context_format": "structured",
              "website": "shop.test"}
    rows = [{"id": "ap1", "task_type": AP, "question_type": "operation", "gold": 0,
             "options": ["CLICK: Click an element.", "TYPE_TEXT: Enter text.", "DONE: Every requirement is met."]},
            {"id": "eg1", "task_type": EG, "question_type": "click_target", "gold": 1,
             "options": ['3: {"element": "[3] Button 3", "current_value": ""}',
                         '5: {"element": "[5] Button 5", "current_value": ""}']}]
    root = tmp_path / "webjev"
    root.mkdir()
    pq.write_table(pa.Table.from_pylist([r | common for r in rows]), root / "mini.parquet")
    return root


def _down(port) -> bool:
    try:
        httpx.get(f"http://127.0.0.1:{port}/v1/models", timeout=2)
    except (httpx.ConnectError, httpx.ConnectTimeout):   # Windows retries a refused localhost connect for ~2 s
        return True
    return False


def test_serve_and_collect_scores_through_the_server(tmp_path):
    port, root = _port(), _webjev(tmp_path)
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"cal": ["ap1"], "test_in": [], "test_shift": ["eg1"]}))
    rows = tmp_path / "out" / "rows.jsonl"
    collect = [sys.executable, str(ROOT / "scripts" / "offline_collect.py"), "--plan", str(plan), "--root", str(root),
               "--base-url", f"http://127.0.0.1:{port}", "--out", str(rows), "--n-perm", "3"]
    card = serve_and_collect(_server(tmp_path, port, "/out/webjev-v1"), port, collect, expect_run="/out/webjev-v1",
                             log=tmp_path / "serve.log", poll=0.1, ready_timeout=60)
    assert card == {"name": "kev-latest", "run": "/out/webjev-v1"}
    got = {r["id"]: r for r in map(json.loads, rows.read_text().splitlines())}
    assert set(got) == {"ap1", "eg1"} and got["ap1"]["split"] == "cal" and got["eg1"]["split"] == "test_shift"
    assert got["ap1"]["probs"] == pytest.approx({"CLICK": 1 / 3, "TYPE_TEXT": 1 / 3, "DONE": 1 / 3})
    assert got["ap1"]["n_perm"] == 3 and got["ap1"]["backend"] == "kev" and got["eg1"]["gold_id"] == "E5"
    assert (tmp_path / "serve.log").exists() and _down(port)


def test_serve_and_collect_refuses_a_fallback_run(tmp_path):
    port, marker = _port(), tmp_path / "collected"
    with pytest.raises(RuntimeError, match="not /out/webjev-v1"):
        serve_and_collect(_server(tmp_path, port, "runs/smoke"), port, [sys.executable, "-c", f"open({str(marker)!r}, 'w')"],
                          expect_run="/out/webjev-v1", poll=0.1, ready_timeout=60)
    assert not marker.exists() and _down(port)


def test_serve_and_collect_reports_a_server_that_exits(tmp_path):
    port = _port()
    with pytest.raises(RuntimeError, match="exited with code 3"):
        serve_and_collect(_server(tmp_path, port, "x", "die"), port, [sys.executable, "-c", "pass"], poll=0.1)


def test_serve_and_collect_times_out_on_a_server_that_never_answers(tmp_path):
    port = _port()
    with pytest.raises(TimeoutError, match="no answer"):
        serve_and_collect(_server(tmp_path, port, "x", "busy"), port, [sys.executable, "-c", "pass"], poll=0.1,
                          ready_timeout=1)
    assert _down(port)


def test_serve_and_collect_stops_the_server_when_collection_fails(tmp_path):
    port = _port()
    with pytest.raises(subprocess.CalledProcessError):
        serve_and_collect(_server(tmp_path, port, "x"), port, [sys.executable, "-c", "raise SystemExit(2)"],
                          expect_run="x", poll=0.1, ready_timeout=60)
    assert _down(port)


def test_score_plan_returns_its_files_resumes_and_refuses_another_run_or_plan(tmp_path):
    root, out = _webjev(tmp_path), tmp_path / "scores" / "m2-x"
    plan = json.dumps({"cal": ["ap1"], "test_in": [], "test_shift": ["eg1"]})
    script = str(ROOT / "scripts" / "offline_collect.py")

    def run(run_id, plan_text, served=None):
        port = _port()
        return score_plan(out, run_id, plan_text, _server(tmp_path, port, served or run_id), script, str(root), port,
                          ["--n-perm", "3"], poll=0.1, ready_timeout=60)

    files = run("/out/a", plan)
    assert set(files) == {"meta.json", "plan.json", "rows.jsonl", "serve.log", "server.json"}
    assert len(files["rows.jsonl"].splitlines()) == 2 and json.loads(files["server.json"])["run"] == "/out/a"
    assert run("/out/a", plan)["rows.jsonl"] == files["rows.jsonl"]   # resumed: nothing re-scored or duplicated
    with pytest.raises(ValueError, match="new name"):
        run("/out/b", plan)
    with pytest.raises(ValueError, match="new name"):
        run("/out/a", json.dumps({"cal": ["ap1"], "test_in": [], "test_shift": []}))
    assert read_files(out)["rows.jsonl"] == files["rows.jsonl"]
