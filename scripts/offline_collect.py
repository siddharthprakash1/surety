import argparse
import json
import sys
from pathlib import Path

from surety.decide import MockBackend, SystemOneClient
from surety.eval.collect import collect
from surety.eval.webjev import load_instances

ap = argparse.ArgumentParser()
ap.add_argument("--plan", default="runs/plan.json")
ap.add_argument("--root", default="data/webjev")
ap.add_argument("--backend", choices=["kev", "clef", "mock"], default="kev")
ap.add_argument("--base-url", default="http://127.0.0.1:8009")
ap.add_argument("--model", default="kev-latest")
ap.add_argument("--mode", choices=["compact", "full"], default="compact")
ap.add_argument("--n-perm", type=int, default=5)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
ap.add_argument("--limit", type=int)
ap.add_argument("--splits", default="cal,test_in,test_shift")
ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
a = ap.parse_args()

plan = json.loads(Path(a.plan).read_text())
split_of = {i: s for s in a.splits.split(",") for i in plan[s]}
by_id = {i.id: i for i in load_instances(Path(a.root))}
ids = list(split_of)[: a.limit]
backend = MockBackend() if a.backend == "mock" else SystemOneClient(a.base_url, a.model, name=a.backend)
n = collect([by_id[i] for i in ids], backend, a.mode, a.n_perm, Path(a.out), split_of, a.seed, a.resume)
print(f"wrote {n} rows to {a.out}", file=sys.stderr)
sys.exit(0 if n else 1)
