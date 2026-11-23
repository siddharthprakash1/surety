"""H4 on MiniWoB++: six systems × tasks × seeds, then the table. Resumable (one record per episode).

    MINIWOB_URL=http://127.0.0.1:8765/miniwob/ uv run python scripts/run_h4.py --backend kev \
        --fit artifacts/m1/kev_compact.jsonl --supervisor-url <endpoint> --supervisor-model <model> --supervisor-json \
        --tasks @configs/miniwob_h4.txt --seeds 0,1,2 --shadow-rate 0.2 --out runs/h4/miniwob.jsonl
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from surety.decide import MockBackend, SystemOneClient
from surety.env import BrowserGymEnv
from surety.eval.analyze import load_collection
from surety.eval.live import ALL_SYSTEMS, SYSTEMS, h4_table, run_episodes
from surety.supervise.cli import add_supervisor_args, make_supervisor


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True, help="comma list of MiniWoB++ task names, or @file with one per line")
    ap.add_argument("--systems", default=",".join(SYSTEMS),
                    help=f"comma list from {','.join(ALL_SYSTEMS)}; default the six H4 systems")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--audit-rate", type=float, default=0.2)
    ap.add_argument("--shadow-rate", type=float, default=0.0,
                    help="evaluation labels on acted steps (not counted as supervisor calls) for live auto_err")
    ap.add_argument("--max-steps", type=int, default=20, help="click-checkboxes-large needs up to 13 steps")
    ap.add_argument("--fit", type=Path, help="offline collection whose cal rows fit T and the gates")
    ap.add_argument("--backend", choices=["mock", "kev"], default="mock")
    ap.add_argument("--base-url", default="http://127.0.0.1:8009")
    ap.add_argument("--n-perm", type=int, help="orderings per step; default 5 (3 above 16 options) with --fit, as fitted, else 1")
    add_supervisor_args(ap)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    unknown = [s for s in a.systems.split(",") if s not in ALL_SYSTEMS]
    if unknown:
        sys.exit(f"unknown systems {unknown}; choose from {sorted(ALL_SYSTEMS)}")
    if not os.environ.get("MINIWOB_URL"):
        sys.exit("MINIWOB_URL is not set (see scripts/run_miniwob.py)")

    tasks = (Path(a.tasks[1:]).read_text().split() if a.tasks.startswith("@") else a.tasks.split(","))
    supervisor = make_supervisor(a)
    # a resume must use the same models and fit: they change every result
    setup = {"backend": a.backend, "base_url": a.base_url if a.backend == "kev" else None,
             "fit": hashlib.sha256(a.fit.read_bytes()).hexdigest()[:16] if a.fit else None,
             "supervisor": {"url": a.supervisor_url, "model": a.supervisor_model, "json": a.supervisor_json,
                            "max_tokens": a.supervisor_max_tokens, "reasoning": a.supervisor_reasoning,
                            "temperature": None if a.supervisor_no_temperature else 0.0,
                            "token_param": a.supervisor_token_param}}
    backend = SystemOneClient(a.base_url) if a.backend == "kev" else MockBackend()
    n = run_episodes(tasks, a.systems.split(","), [int(s) for s in a.seeds.split(",")],
                     lambda t: BrowserGymEnv("miniwob." + t), backend, supervisor, a.out,
                     rows=load_collection(a.fit) if a.fit else None, eps=a.eps, audit_rate=a.audit_rate,
                     max_steps=a.max_steps, n_perm=a.n_perm or (5 if a.fit else 1),
                     n_perm_large=3 if a.fit and not a.n_perm else None, shadow_rate=a.shadow_rate, setup=setup)
    records = [json.loads(x) for x in a.out.read_text().splitlines() if x]
    report = a.out.with_suffix(".md")
    report.write_text(f"# H4: {a.out.stem}\n\n- episodes: {len(records)} (new {n}); eps {a.eps}; audit rate "
                      f"{a.audit_rate} (arisk); shadow rate {a.shadow_rate}\n\n{h4_table(records)}", encoding="utf-8")
    print(f"{n} new episodes; table in {report}")  # the table has non-ASCII (Δ): a cp1252 console cannot print it
    return 0


if __name__ == "__main__":
    sys.exit(main())
