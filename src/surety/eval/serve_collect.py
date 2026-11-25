"""Start a scoring server, wait until it answers, run a collection against it, stop it. The body of the Modal scoring job
(scripts/modal_score_kev.py), kept here so it runs and is tested without Modal."""
import os
import subprocess
import time
from pathlib import Path

import httpx


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
