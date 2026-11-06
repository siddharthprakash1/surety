import argparse
import json
from pathlib import Path

from surety.eval.webjev import load_instances, make_plan, site_split

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/webjev")
ap.add_argument("--n-cal", type=int, default=1500)
ap.add_argument("--n-test-in", type=int, default=1000)
ap.add_argument("--n-test-shift", type=int, default=1500)
ap.add_argument("--holdout-frac", type=float, default=0.3)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="runs/plan.json")
a = ap.parse_args()

insts = load_instances(Path(a.root))
cal_sites, held_sites = site_split(insts, a.holdout_frac, a.seed)
plan = make_plan(insts, cal_sites, held_sites, a.n_cal, a.n_test_in, a.n_test_shift, a.seed)
out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"seed": a.seed, "cal_sites": sorted(cal_sites), "held_sites": sorted(held_sites), **plan}))
print({k: len(v) for k, v in plan.items()})
