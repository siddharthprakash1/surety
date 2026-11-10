import argparse
import json
import sys
from pathlib import Path

from surety.eval.analyze import _split, load_collection, select_escalated, variants
from surety.eval.webjev import compile_instance, load_instances
from surety.gate import SplitConformal
from surety.supervise.client import Supervisor


def run(pairs, by_id, sup, out: Path, log=sys.stderr) -> None:
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # half-written last line
            done.add((r["id"], r["cond"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as f:
        for k, (r, c) in enumerate(pairs, 1):
            state = compile_instance(by_id[r["id"]], "compact")
            for cond, ids in (("full", None), ("set", [o.id for o in state.options if o.id in c])):
                if (r["id"], cond) in done:
                    continue
                try:
                    chosen, u = sup.choose(state, ids)
                except Exception as e:
                    print(f"skip {r['id']} {cond}: {e!r}", file=log)
                    continue
                f.write(json.dumps({
                    "id": r["id"], "split": r["split"], "task_type": r["task_type"], "cond": cond,
                    "set_size": len(c), "n_options": len(r["option_ids"]), "gold_id": r["gold_id"],
                    "chosen": chosen, "correct": chosen == r["gold_id"], "gold_in_set": r["gold_id"] in c,
                    "prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens,
                    "latency_ms": u.latency_ms, "fallback_parsed": u.fallback_parsed,
                    "abstained": u.abstained}) + "\n")
                f.flush()
            if k % 10 == 0:
                print(f"{k}/{len(pairs)}", file=log)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=Path("data/webjev"))
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--splits", default="test_in,test_shift")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base-url", default="http://127.0.0.1:11434")
    ap.add_argument("--model", default="gpt-oss:20b")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--reasoning", choices=["low", "medium", "high"])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    rows = load_collection(a.collect)
    probs = variants(rows)["debiased+T"]
    policy = SplitConformal(a.eps)
    policy.fit(*_split(rows, probs, "cal"))
    pairs = select_escalated(rows, policy, a.splits.split(","), a.n, a.seed, probs)
    by_id = {i.id: i for i in load_instances(a.root)}
    print(f"{len(pairs)} escalated rows, q={policy.q:.3f}", file=sys.stderr)
    run(pairs, by_id, Supervisor(a.base_url, a.model, max_tokens=a.max_tokens, reasoning=a.reasoning), a.out)


if __name__ == "__main__":
    main()
