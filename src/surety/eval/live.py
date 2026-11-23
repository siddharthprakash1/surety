"""H4: systems × tasks × seeds on a live env; one JSONL record per episode, resumable.

H4 (spec): end to end, within 2 points of supervisor-only success at ≤ 35 % of its supervisor calls, compared on the
(seed, task) pairs both ran. Live auto_err (est.) = acted share × disagreement rate on the labelled acted steps (audits
and shadow labels, one random sample), relative to the supervisor; shadow labels are evaluation calls, not counted as
the system's supervisor calls. Each system keeps one gate for its whole stream (seed-major, then task), so adaptive gates
carry α across episodes — also across resumes, which restore α and the random streams from the file.
"""
import json
import random
from pathlib import Path

import numpy as np

from surety.eval.analyze import _table
from surety.gate import CriticalDetector, CriticalOverride
from surety.loop import run
from surety.loop.fit import fit_gate
from surety.trace import Trace

SYSTEMS = {"supervisor": "supervisor", "executor": "executor", "fixed": "fixed:0.7",
           "escalate_on_none": "escalate_on_none", "risk": "risk:{eps}", "arisk": "arisk:{eps}"}
MAX_ERRORS = 3  # consecutive failed episodes stop the run (a dead server), rather than record a stream of failures
_EMPTY = {"reward": 0.0, "steps": 0, "executor_calls": 0, "supervisor_calls": 0, "acted": 0, "labeled": 0,
          "label_wrong": 0, "shadow_calls": 0, "choose_calls": 0, "unanswered": 0, "critical": []}


def _load(out: Path) -> list[dict]:
    if not out.exists():
        return []
    raw = out.read_bytes()
    if raw and not raw.endswith(b"\n"):  # half-written last line
        raw = raw[: raw.rfind(b"\n") + 1]
        out.write_bytes(raw)
    return [json.loads(x) for x in raw.decode().splitlines() if x.strip()]


def _random(state: list) -> random.Random:
    r = random.Random()
    r.setstate((state[0], tuple(state[1]), state[2]))
    return r


def _labels(steps: list[dict]) -> dict:
    """Acted steps with a supervisor label (audit or shadow: one random sample) and how many disagree."""
    got = [(s.get("audit", s.get("shadow")), s["candidate_set"][0]) for s in steps
           if s["decision"] == "act" and ("audit" in s or "shadow" in s)]
    calls = [c for s in steps for c in s["supervisor"] if c["call"] == "choose"]
    return {"labeled": sum(lab is not None for lab, _ in got), "label_wrong": sum(lab not in (None, o) for lab, o in got),
            "choose_calls": len(calls), "unanswered": sum(not c.get("answered", True) for c in calls)}


def run_episodes(tasks, systems, seeds, make_env, backend, supervisor, out: Path, rows=None, eps: float = 0.05,
                 audit_rate: float = 0.2, max_steps: int = 10, n_perm: int = 1, n_perm_large: int | None = None,
                 shadow_rate: float = 0.0, setup: dict | None = None) -> int:
    """make_env(task) -> env with reset(seed) -> goal, observe/act, reward, close. Returns episodes written.
    Step traces go to <out stem>_traces/<system>-<seed>-<task>.jsonl (a retried episode replaces its trace).
    setup: what else changes results (backend, fit collection, supervisor); a resume must match it and the settings.
    stop "error:<Type>" = the episode could not run (retried on resume); the runner's own "error" (an action the page
    rejected) is an ordinary failed episode."""
    out.parent.mkdir(parents=True, exist_ok=True)
    traces = out.parent / f"{out.stem}_traces"
    settings = {"eps": eps, "audit_rate": audit_rate, "shadow_rate": shadow_rate, "max_steps": max_steps,
                "n_perm": n_perm, "n_perm_large": n_perm_large, **(setup or {})}
    old = _load(out)
    if bad := next((r for r in old if {k: r.get(k) for k in settings} != settings), None):
        raise ValueError(f"{out} holds episodes run with {({k: bad.get(k) for k in settings})}; use another --out")
    done = {(r["system"], r["seed"], r["task"]) for r in old if not r["stop"].startswith("error:")}
    alpha = {r["system"]: r["alpha_raw"] for r in old if r.get("alpha_raw") is not None}  # file order = stream order
    streams = {(r["system"], r["seed"]): (r["audit_rng"], r["shadow_rng"]) for r in old}
    del old  # each record carries two Mersenne Twister states (~15 KB); keep only the last per stream
    written = errors = 0
    with out.open("a") as f:
        for system in systems:
            spec = SYSTEMS[system].format(eps=eps)
            gate, scaler = fit_gate(spec, rows if ":" in spec else None)
            if system in alpha and hasattr(gate, "alpha_raw"):
                gate.alpha_raw = alpha[system]
            policy = CriticalOverride(gate, CriticalDetector())
            for seed in seeds:
                if (system, seed) in streams:
                    rng, shadow_rng = map(_random, streams[(system, seed)])
                else:
                    rng, shadow_rng = random.Random(f"audit-{system}-{seed}"), random.Random(f"shadow-{system}-{seed}")
                for task in tasks:
                    if (system, seed, task) in done:
                        continue
                    path = traces / f"{system}-{seed}-{task}.jsonl"
                    path.unlink(missing_ok=True)
                    rec, env, trace = {"system": system, "spec": spec, "seed": seed, "task": task, **settings}, None, None
                    try:
                        env = make_env(task)
                        trace = Trace(path)
                        goal = env.reset(seed=seed)
                        ep = run(goal, env, backend, policy, supervisor, trace, max_steps=max_steps, n_perm=n_perm,
                                 n_perm_large=n_perm_large, calibrate=scaler.apply if scaler else None,
                                 audit_rate=audit_rate if system == "arisk" else 0.0, rng=rng,
                                 shadow_rate=shadow_rate, shadow_rng=shadow_rng)
                        rec.update(reward=float(env.reward), stop=ep.stop_reason, steps=len(ep.steps),
                                   executor_calls=ep.executor_calls, supervisor_calls=ep.supervisor_calls,
                                   acted=sum(s["decision"] == "act" for s in ep.steps), **_labels(ep.steps),
                                   shadow_calls=ep.shadow_calls,
                                   critical=[s["critical"] for s in ep.steps if "critical" in s])
                        errors = 0
                    except Exception as e:  # recorded, retried on resume
                        errors += 1
                        rec.update(_EMPTY, stop=f"error:{type(e).__name__}", error=str(e)[:300])
                    finally:
                        for closable in (trace, env):
                            try:
                                if closable:
                                    closable.close()
                            except Exception:  # a failing close must not lose the episode's record
                                pass
                    rec.update(alpha_raw=getattr(gate, "alpha_raw", None), audit_rng=rng.getstate(),
                               shadow_rng=shadow_rng.getstate())
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    written += 1
                    if errors >= MAX_ERRORS:
                        raise RuntimeError(f"{errors} consecutive episodes failed, last: {rec['error']}")
    return written


def h4_table(records: list[dict]) -> str:
    ok = [r for r in records if not r["stop"].startswith("error:")]
    sup = {(r["seed"], r["task"]): r for r in ok if r["system"] == "supervisor"}
    body = []
    for system in [s for s in SYSTEMS if any(r["system"] == s for r in records)]:
        rs = [r for r in ok if r["system"] == system]
        errs = sum(r["system"] == system and r["stop"].startswith("error:") for r in records)
        if not rs:
            body.append([system, "-", 0, "-", "-", "-", "-", "-", "-", "-", "-", errs, "-"])
            continue
        succ, calls = np.mean([r["reward"] > 0 for r in rs]), np.mean([r["supervisor_calls"] for r in rs])
        acted = sum(r["acted"] for r in rs) / max(sum(r["steps"] for r in rs), 1)
        pairs = [(r, sup[(r["seed"], r["task"])]) for r in rs if (r["seed"], r["task"]) in sup]
        if pairs and system != "supervisor":
            d_succ = float(np.mean([a["reward"] > 0 for a, _ in pairs]) - np.mean([b["reward"] > 0 for _, b in pairs]))
            b_calls = np.mean([b["supervisor_calls"] for _, b in pairs])
            ratio = float(np.mean([a["supervisor_calls"] for a, _ in pairs]) / b_calls) if b_calls else float("nan")
            h4 = str(d_succ >= -0.02 - 1e-9 and ratio <= 0.35 + 1e-9)
        else:
            d_succ = ratio = h4 = "-"
        n_lab = sum(r.get("labeled", 0) for r in rs)
        live_err = f"{acted * sum(r.get('label_wrong', 0) for r in rs) / n_lab:.3f} (n={n_lab})" if n_lab else "-"
        n_choose = sum(r.get("choose_calls", 0) for r in rs)
        body.append([system, rs[0]["spec"], len(rs), float(succ), d_succ, len(pairs), float(calls), ratio,
                     float(acted), live_err, f"{sum(r.get('unanswered', 0) for r in rs) / n_choose:.3f}" if n_choose else "-",
                     errs, h4])
    return _table(("system", "gate", "episodes", "success", "Δ vs supervisor (matched)", "matched", "supervisor_calls/ep",
                   "calls vs supervisor (matched)", "acted share", "live auto_err (est.)", "unanswered calls",
                   "errors", "H4"), body)
