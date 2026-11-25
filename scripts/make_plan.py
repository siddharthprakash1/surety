import argparse
import json
from pathlib import Path

from surety.eval.sites import psl_version
from surety.eval.webjev import load_instances, make_plan, make_spread_plan, site_split, site_split3

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/webjev")
ap.add_argument("--design", choices=["m1", "spread"], default="m1",
                help="m1: calibration and held-out hosts, steps drawn uniformly (M1). spread: training, calibration and "
                     "held-out websites (hosts grouped by surety.eval.sites.website), each split spread over its "
                     "websites (M2, for a fine-tuned executor)")
ap.add_argument("--n-cal", type=int, default=1500)
ap.add_argument("--n-test-in", type=int, default=1000)
ap.add_argument("--n-test-shift", type=int, default=1500)
ap.add_argument("--holdout-frac", type=float, default=0.3, help="m1: share of sites held out")
ap.add_argument("--cal-frac", type=float, default=0.25, help="spread: share of sites for cal and test_in")
ap.add_argument("--test-frac", type=float, default=0.25, help="spread: share of sites for test_shift")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max-options", type=int)
ap.add_argument("--out", default="runs/plan.json")
a = ap.parse_args()

insts = load_instances(Path(a.root))
if a.max_options:
    insts = [i for i in insts if len(i.options) <= a.max_options]
if a.design == "m1":
    cal_sites, held_sites = site_split(insts, a.holdout_frac, a.seed)
    plan = make_plan(insts, cal_sites, held_sites, a.n_cal, a.n_test_in, a.n_test_shift, a.seed)
    head = {"seed": a.seed, "max_options": a.max_options, "cal_sites": sorted(cal_sites), "held_sites": sorted(held_sites)}
else:
    train_sites, cal_sites, held_sites = site_split3(insts, a.cal_frac, a.test_frac, a.seed)
    plan = make_spread_plan(insts, cal_sites, held_sites, a.n_cal, a.n_test_in, a.n_test_shift, a.seed)
    head = {"seed": a.seed, "max_options": a.max_options, "design": "spread",
            "site_key": f"website: registrable domain without its public suffix (publicsuffixlist {psl_version()})",
            "train_sites": sorted(train_sites), "cal_sites": sorted(cal_sites), "held_sites": sorted(held_sites)}
out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({**head, **plan}))
print({k: len(v) for k, v in plan.items()})
