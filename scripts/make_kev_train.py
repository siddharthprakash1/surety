import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from surety.eval.kev_train import build_records, write_jsonl
from surety.eval.webjev import load_instances

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/webjev")
ap.add_argument("--plan", default="runs/plan.json")
ap.add_argument("--n-train", type=int, default=8000)
ap.add_argument("--n-heldout", type=int, default=500)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max-options", type=int, default=64)
ap.add_argument("--out-dir", default="runs/kev_ft")
a = ap.parse_args()

insts = load_instances(Path(a.root))
site_of = {i.id: i.site for i in insts}
train, heldout = build_records(insts, json.loads(Path(a.plan).read_text()), a.n_train, a.n_heldout, a.seed,
                               a.max_options)


def stats(records):
    n_opts = [len(q["criteria"]) for r in records for q in r["questions"].values()]
    q = statistics.quantiles(n_opts, n=100) if len(n_opts) > 1 else n_opts * 99
    return {"n": len(records), "by_task_type": dict(Counter(qid for r in records for qid in r["questions"])),
            "sites": len({site_of[r["_meta"]["id"]] for r in records}),
            "mean_state_chars": round(statistics.fmean(len(r["state"]) for r in records), 1) if records else 0,
            "options": {"p50": q[49], "p90": q[89], "p99": q[98], "max": max(n_opts, default=0)}}


out = Path(a.out_dir)
write_jsonl(train, out / "train.jsonl")
write_jsonl(heldout, out / "heldout.jsonl")
summary = {"args": vars(a), "train": stats(train), "heldout": stats(heldout)}
(out / "stats.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
