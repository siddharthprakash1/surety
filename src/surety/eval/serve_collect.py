"""Start a scoring server, wait until it answers, run a collection against it, stop it. The body of the Modal scoring job
(scripts/modal_score_kev.py), kept here so it runs and is tested without Modal."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

from surety.eval.webjev import plan_digest


def wait_ready(url: str, proc: subprocess.Popen, timeout: float, poll: float = 5.0) -> dict:
    """GET url until it answers 200 (-> its JSON). Raises if the server exits first or `timeout` seconds pass."""
    deadline = time.monotonic() + timeout
    while True:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with code {proc.returncode} before answering {url}")
        try:
            r = httpx.get(url, timeout=10)
            if r.status_code == 200:
                return r.json()
        except httpx.HTTPError:
            pass
        if time.monotonic() >= deadline:
            raise TimeoutError(f"no answer from {url} after {timeout:.0f} s")
        time.sleep(poll)


def serve_and_collect(server_cmd: list[str], port: int, collect_cmd: list[str], expect_run: str | None = None,
                      log: Path | None = None, env: dict | None = None, cwd: str | None = None,
                      ready_timeout: float = 1800, poll: float = 5.0) -> dict:
    """Run server_cmd, check that /v1/models serves expect_run (kev.serve falls back to another run when the requested
    one is missing), run collect_cmd to completion, then stop the server. -> the served model card."""
    with open(log or os.devnull, "w", encoding="utf-8") as out:
        proc = subprocess.Popen(server_cmd, stdout=out, stderr=subprocess.STDOUT, env=env, cwd=cwd)
        try:
            cards = wait_ready(f"http://127.0.0.1:{port}/v1/models", proc, ready_timeout, poll).get("models") or [{}]
            served = {str(c.get("run")) for c in cards}
            if expect_run is not None and served != {expect_run}:
                raise RuntimeError(f"server runs {sorted(served)}, not {expect_run}")
            subprocess.run(collect_cmd, check=True, env=env, cwd=cwd)
            return cards[0]
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def read_files(out: Path) -> dict[str, str]:
    """A scoring directory's files by name: rows.jsonl, server.json, serve.log, plan.json, meta.json."""
    return {p.name: p.read_text(encoding="utf-8", errors="replace") for p in sorted(Path(out).iterdir()) if p.is_file()}


def score_plan(out: Path, run: str, plan: str, server_cmd: list[str], collect_script: str, root: str, port: int,
               extra: list[str] | tuple = (), env: dict | None = None, cwd: str | None = None,
               ready_timeout: float = 1800, poll: float = 5.0) -> dict[str, str]:
    """One scoring job in `out`. Refuses a directory already scored with another run or plan (offline_collect would skip
    its rows and they would pass as this run's), collects the plan into out/rows.jsonl through server_cmd, which must
    serve `run`, and returns the directory's files for the caller to save."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"run": run, "plan_sha256": plan_digest(json.loads(plan))}
    if (out / "meta.json").exists():
        old = json.loads((out / "meta.json").read_text())
        if old != meta:
            raise ValueError(f"{out} holds rows of run {old['run']} on plan {old['plan_sha256'][:12]}, not run {run} "
                             f"on plan {meta['plan_sha256'][:12]}: score under a new name")
    (out / "meta.json").write_text(json.dumps(meta))
    (out / "plan.json").write_text(plan)
    collect = [sys.executable, collect_script, "--plan", str(out / "plan.json"), "--root", root,
               "--base-url", f"http://127.0.0.1:{port}", "--out", str(out / "rows.jsonl"), *extra]
    card = serve_and_collect(server_cmd, port, collect, expect_run=run, log=out / "serve.log", env=env, cwd=cwd,
                             ready_timeout=ready_timeout, poll=poll)
    (out / "server.json").write_text(json.dumps(card, indent=1))
    return read_files(out)
