"""H3: the supervisor on escalated rows, full option list vs the conformal set. Resumable: re-running the same command
retries failed calls and continues after a rate-limit stop (exit 75), so a free API tier can finish it over days."""
import argparse
import json
import sys
from pathlib import Path

from surety.eval.analyze import _split, load_collection, select_escalated, variants
from surety.eval.webjev import compile_instance, load_instances
from surety.gate import SplitConformal
from surety.supervise.cli import api_key, parse_params
from surety.supervise.client import RateLimited, Supervisor


def run(pairs, by_id, sup, out: Path, log=sys.stderr, setup: dict | None = None) -> bool:
    """setup: what changes the answers (model, mode, reasoning, max tokens, eps); a resume must match it exactly.
    Returns False when it stopped at a rate limit."""
    setup = setup or {"model": sup.model, "json_mode": sup.json_mode}
    done = set()
    if out.exists():
        raw = out.read_bytes()
        if raw and not raw.endswith(b"\n"):  # half-written last line: drop it, or the next record is appended onto it
            raw = raw[: raw.rfind(b"\n") + 1]
            out.write_bytes(raw)
        for line in raw.decode().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (theirs := {k: r.get(k) for k in setup}) != setup:
                sys.exit(f"{out} holds rows from another setup {theirs}; use another --out")
            done.add((r["id"], r["cond"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    total, failed = 2 * len(pairs), 0
    saved = sum((r["id"], cond) in done for r, _ in pairs for cond in ("full", "set"))
    with open(out, "a") as f:
        for k, (r, c) in enumerate(pairs, 1):
            state = compile_instance(by_id[r["id"]], "compact")
            for cond, ids in (("full", None), ("set", [o.id for o in state.options if o.id in c])):
                if (r["id"], cond) in done:
                    continue
                try:
                    chosen, u = sup.choose(state, ids)
                except RateLimited as e:  # a daily cap: every next call would fail too
                    print(f"stopped: {e}; {saved}/{total} calls saved, re-run the same command later", file=log)
                    return False
                except Exception as e:
                    print(f"skip {r['id']} {cond}: {e!r}", file=log)
                    failed += 1
                    continue
                f.write(json.dumps({
                    "id": r["id"], "split": r["split"], "task_type": r["task_type"], "cond": cond,
                    "set_size": len(c), "n_options": len(r["option_ids"]), "gold_id": r["gold_id"],
                    "chosen": chosen, "correct": chosen == r["gold_id"], "gold_in_set": r["gold_id"] in c,
                    "prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens,
                    "latency_ms": u.latency_ms, "fallback_parsed": u.fallback_parsed,
                    "abstained": u.abstained, **setup}) + "\n")
                f.flush()
                saved += 1
            if k % 10 == 0:
                print(f"{k}/{len(pairs)}", file=log)
    print(f"{saved}/{total} calls saved" + (f"; {failed} failed, re-run to retry them" if failed else ""), file=log)
    return True


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
    ap.add_argument("--json", action="store_true", help="constrain replies to {\"id\": <option id> | NONE}")
    ap.add_argument("--api-key-env", help="environment variable (or ./.env entry) holding the endpoint's API key")
    ap.add_argument("--no-temperature", action="store_true", help="omit temperature (OpenAI reasoning models reject it)")
    ap.add_argument("--token-param", choices=["max_tokens", "max_completion_tokens"], default="max_tokens")
    ap.add_argument("--param", action="append", metavar="KEY=VALUE",
                    help="extra request field, repeatable: reasoning_effort=low for Groq's gpt-oss")
    ap.add_argument("--max-wait", type=float, default=900.0,
                    help="seconds a call may wait out rate limits; a longer wait (a daily cap) stops the run")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    key, params = api_key(a.api_key_env), parse_params(a.param)

    rows = load_collection(a.collect)
    probs = variants(rows)["debiased+T"]
    policy = SplitConformal(a.eps)
    policy.fit(*_split(rows, probs, "cal"))
    pairs = select_escalated(rows, policy, a.splits.split(","), a.n, a.seed, probs)
    by_id = {i.id: i for i in load_instances(a.root)}
    print(f"{len(pairs)} escalated rows, q={policy.q:.3f}", file=sys.stderr)
    sup = Supervisor(a.base_url, a.model, max_tokens=a.max_tokens, reasoning=a.reasoning, api_key=key, json_mode=a.json,
                     temperature=None if a.no_temperature else 0.0, token_param=a.token_param, params=params,
                     max_wait=a.max_wait)
    # params is None unless given, as in rows written before it existed, so those files still resume
    setup = {"model": a.model, "json_mode": a.json, "reasoning": a.reasoning, "max_tokens": a.max_tokens, "eps": a.eps,
             "params": params}
    if not run(pairs, by_id, sup, a.out, setup=setup):
        sys.exit(75)  # EX_TEMPFAIL: try again later


if __name__ == "__main__":
    main()
