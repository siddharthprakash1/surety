import json
import sys
import time
from pathlib import Path

import httpx

from surety.decide.base import Backend
from surety.eval.webjev import compile_instance
from surety.types import Instance

MAX_CONSECUTIVE_FAILURES = 20


class CollectAborted(RuntimeError):
    pass


def _load_done(out: Path, mode: str, backend: str) -> set[tuple]:
    if not out.exists():
        return set()
    raw = out.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raw = raw[: raw.rfind(b"\n") + 1]
        out.write_bytes(raw)
    done, other = set(), set()
    for line in raw.decode().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        (done if (r["mode"], r["backend"]) == (mode, backend) else other).add((r["id"], r["mode"], r["backend"]))
    if other:
        print(f"warning: {len(other)} existing rows have a different mode/backend than ({mode}, {backend})",
              file=sys.stderr)
    return done | other


def collect(instances: list[Instance], backend: Backend, mode: str, n_perm: int, out: Path,
            split_of: dict[str, str], seed: int = 0, resume: bool = True,
            n_perm_large: int | None = None, large_options: int = 16) -> int:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not resume:
        out.write_text("")
    done = _load_done(out, mode, backend.name)
    written = skipped = streak = 0
    lat, t0 = 0.0, time.time()
    with out.open("a") as f:
        for inst in instances:
            key = (inst.id, mode, backend.name)
            if key in done:
                continue
            state = compile_instance(inst, mode)
            k = n_perm_large if n_perm_large is not None and len(state.options) > large_options else n_perm
            try:
                ans = backend.choice(state, n_perm=k, seed=seed)
            except httpx.HTTPError as e:
                skipped += 1
                streak += 1
                print(f"skip {inst.id}: {e!r}", file=sys.stderr)
                if streak >= MAX_CONSECUTIVE_FAILURES:
                    raise CollectAborted(f"{streak} consecutive failures; written {written}, skipped {skipped}")
                continue
            streak = 0
            f.write(json.dumps({
                "id": inst.id, "split": split_of[inst.id], "site": inst.site, "task_type": inst.task_type,
                "question_type": inst.question_type, "mode": mode, "backend": backend.name,
                "gold_id": state.meta["gold_id"], "option_ids": [o.id for o in state.options],
                "n_options": len(state.options), "state_chars": len(state.state_text), "n_perm": k,
                "probs": ans.probs, "runs": ans.runs, "latency_ms": ans.latency_ms,
                "input_tokens": ans.input_tokens, "truncated": ans.truncated}) + "\n")
            f.flush()
            done.add(key)
            written += 1
            lat += ans.latency_ms
            if (written + skipped) % 25 == 0:
                print(f"{written} rows, {skipped} skipped, mean latency {lat / max(written, 1):.0f} ms, "
                      f"{time.time() - t0:.0f}s", file=sys.stderr)
    print(f"written {written}, skipped {skipped}", file=sys.stderr)
    return written
