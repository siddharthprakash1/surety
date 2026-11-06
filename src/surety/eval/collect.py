import json
import sys
import time
from pathlib import Path

import httpx

from surety.decide.base import Backend
from surety.eval.webjev import compile_instance
from surety.types import Instance


def collect(instances: list[Instance], backend: Backend, mode: str, n_perm: int, out: Path,
            split_of: dict[str, str], seed: int = 0, resume: bool = True) -> int:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not resume:
        out.write_text("")
    done = {json.loads(line)["id"] for line in out.read_text().splitlines() if line.strip()} if out.exists() else set()
    written, lat, t0 = 0, 0.0, time.time()
    with out.open("a") as f:
        for inst in instances:
            if inst.id in done:
                continue
            state = compile_instance(inst, mode)
            try:
                ans = backend.choice(state, n_perm=n_perm, seed=seed)
            except httpx.HTTPError as e:
                print(f"skip {inst.id}: {e!r}", file=sys.stderr)
                continue
            f.write(json.dumps({
                "id": inst.id, "split": split_of[inst.id], "site": inst.site, "task_type": inst.task_type,
                "question_type": inst.question_type, "mode": mode, "backend": backend.name,
                "gold_id": state.meta["gold_id"], "option_ids": [o.id for o in state.options],
                "n_options": len(state.options), "state_chars": len(state.state_text),
                "probs": ans.probs, "runs": ans.runs, "latency_ms": ans.latency_ms,
                "input_tokens": ans.input_tokens, "truncated": ans.truncated}) + "\n")
            f.flush()
            written += 1
            lat += ans.latency_ms
            if written % 25 == 0:
                print(f"{written} rows, mean latency {lat / written:.0f} ms, {time.time() - t0:.0f}s",
                      file=sys.stderr)
    return written
