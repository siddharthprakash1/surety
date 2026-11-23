import argparse
import os
import random
import sys
import time
from pathlib import Path

from surety.decide import MockBackend, SystemOneClient
from surety.env import BrowserGymEnv
from surety.eval.analyze import load_collection
from surety.gate import CriticalDetector, CriticalOverride
from surety.loop import run
from surety.loop.fit import fit_gate
from surety.supervise.cli import add_supervisor_args, make_supervisor
from surety.trace import Trace

SETUP = """MINIWOB_URL is not set. Serve MiniWoB++ first:
  git clone https://github.com/Farama-Foundation/miniwob-plusplus && cd miniwob-plusplus/miniwob/html && python -m http.server 8765
  export MINIWOB_URL=http://localhost:8765/miniwob/"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="click-test")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--backend", choices=["mock", "kev"], default="mock")
    ap.add_argument("--base-url", default="http://127.0.0.1:8009")
    ap.add_argument("--n-perm", type=int, help="orderings per step; default 5 (3 above 16 options) with --fit, as fitted, else 1")
    ap.add_argument("--policy", default="fixed:0.7",
                    help="supervisor | executor | escalate_on_none | fixed:<tau> | margin:<d> | "
                         "fitted|conformal|aci|risk|arisk|learned|learned_pac|alearned:<eps>")
    ap.add_argument("--fit", type=Path, help="offline collection whose cal rows fit T and the gate (e.g. "
                                             "artifacts/m1/kev_compact.jsonl)")
    ap.add_argument("--audit-rate", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    add_supervisor_args(ap)
    args = ap.parse_args()
    if not os.environ.get("MINIWOB_URL"):
        print(SETUP, file=sys.stderr)
        return 2

    gate, scaler = fit_gate(args.policy, load_collection(args.fit) if args.fit else None)
    n_perm = args.n_perm or (5 if args.fit else 1)
    n_perm_large = 3 if args.fit and not args.n_perm else None
    policy = CriticalOverride(gate, CriticalDetector())
    backend = SystemOneClient(args.base_url) if args.backend == "kev" else MockBackend()
    supervisor = make_supervisor(args)
    env = BrowserGymEnv("miniwob." + args.task)
    trace = Trace(Path("runs/miniwob") / f"{args.task}-{int(time.time())}.jsonl")
    try:
        goal = env.reset(seed=args.seed)
        ep = run(goal, env, backend, policy, supervisor, trace, max_steps=args.steps, n_perm=n_perm,
                 n_perm_large=n_perm_large, calibrate=scaler.apply if scaler else None, audit_rate=args.audit_rate,
                 rng=random.Random(f"audit-{args.seed}"))
    finally:
        trace.close()
        env.close()
    print(f"task={ep.task!r} policy={gate.name} stop={ep.stop_reason} steps={len(ep.steps)} "
          f"executor_calls={ep.executor_calls} supervisor_calls={ep.supervisor_calls} reward={env.reward} "
          f"trace={trace.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
