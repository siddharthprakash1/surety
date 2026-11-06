import argparse
import os
import sys
import time
from pathlib import Path

from surety.decide import MockBackend, SystemOneClient
from surety.env import BrowserGymEnv
from surety.gate import CriticalDetector, CriticalOverride, FixedThreshold, SplitConformal
from surety.loop import run
from surety.supervise import MockSupervisor, Supervisor
from surety.trace import Trace

SETUP = """MINIWOB_URL is not set. Serve MiniWoB++ first:
  git clone https://github.com/Farama-Foundation/miniwob-plusplus && cd miniwob-plusplus/miniwob/html && python -m http.server 8080
  export MINIWOB_URL=http://localhost:8080/miniwob/"""
POLICIES = {"fixed": FixedThreshold, "conformal": SplitConformal}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="click-test")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--backend", choices=["mock", "kev"], default="mock")
    ap.add_argument("--base-url", default="http://127.0.0.1:8009")
    ap.add_argument("--policy", default="fixed:0.7", help="fixed:<tau> | conformal:<eps>")
    ap.add_argument("--supervisor-url", help="OpenAI-compatible server; omitted = supervisor answers NONE")
    ap.add_argument("--supervisor-model", default="default")
    args = ap.parse_args()
    if not os.environ.get("MINIWOB_URL"):
        print(SETUP, file=sys.stderr)
        return 2

    kind, value = args.policy.split(":")
    policy = CriticalOverride(POLICIES[kind](float(value)), CriticalDetector())
    backend = SystemOneClient(args.base_url) if args.backend == "kev" else MockBackend()
    supervisor = Supervisor(args.supervisor_url, args.supervisor_model) if args.supervisor_url else MockSupervisor(None)
    env = BrowserGymEnv("miniwob." + args.task)
    trace = Trace(Path("runs/miniwob") / f"{args.task}-{int(time.time())}.jsonl")
    try:
        goal = env.reset()
        ep = run(goal, env, backend, policy, supervisor, trace, max_steps=args.steps)
    finally:
        trace.close()
        env.close()
    print(f"task={ep.task!r} stop={ep.stop_reason} steps={len(ep.steps)} executor_calls={ep.executor_calls} "
          f"supervisor_calls={ep.supervisor_calls} reward={env.reward} trace={trace.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
