"""Uniform vs risk-targeted audits for the adaptive gates at the same audit budget (CPU, ~30 min with 2 processes).

Settings are chosen on calibration-site streams only; held-out streams (M1's test_shift) report every arm.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from surety.eval.analyze import load_collection
from surety.eval.audits import alearned, arisk, audit_streams, mean_of, pick, streams_table, targeted
from surety.gate.audit import AuditPolicy

GATES = {"arisk (γ 0.02)": (arisk, "ipw"), "alearned (κ 0)": (alearned(0.0), "dr"),
         "alearned (κ 0.2, default)": (alearned(0.2), "dr")}
FLOORS = (0.05, 0.1)
MARGINS = (0.0, 0.05, 0.1, 0.15, 0.2)
REFERENCE = "alearned (κ 0.2, default), uniform"


def flat(rate: float):
    return lambda _: AuditPolicy(rate)


mean = mean_of


def run_eps(args: tuple) -> str:
    collect, eps, rate, seeds = args
    rows = load_collection(collect)
    # 1. the same gates with uniform audits, targeted audits, and uniform audits at the targeted arm's realised rate
    configs = {}
    for g, (make, feedback) in GATES.items():
        configs[f"{g}, uniform"] = (make, flat(rate))
        configs.update({f"{g}, targeted (floor {f:g})": (make, targeted(f, feedback)) for f in FLOORS})
    held = audit_streams(rows, eps, configs, "held_out", rate, seeds)
    matched = {f"{g}, uniform at {mean(held, f'{g}, targeted (floor {f:g})', 'audit_rate'):.3f}":
               (make, flat(mean(held, f"{g}, targeted (floor {f:g})", "audit_rate")))
               for g, (make, _) in GATES.items() for f in FLOORS}
    held += audit_streams(rows, eps, matched, "held_out", rate, seeds)
    order = [n for g in GATES for n in dict.fromkeys(r["config"] for r in held) if n.startswith(g + ",")]
    held.sort(key=lambda r: order.index(r["config"]))
    # 2. floor and margin for targeted audits, chosen on calibration-site streams: the most automation among the
    #    candidates whose share of streams over eps is no higher than the default's (uniform audits, κ 0.2)
    cands = {f"alearned (κ {k:g}), targeted (floor {f:g})": (alearned(k), targeted(f)) for f in FLOORS for k in MARGINS}
    cal = audit_streams(rows, eps, {REFERENCE: (alearned(0.2), flat(rate)), **cands}, "calibration", rate, seeds)
    chosen = pick(cal, REFERENCE, cands)
    # 3. the pick on held-out streams, next to uniform audits at its margin
    parts = [f"## eps = {eps}",
             f"### Held-out streams ({seeds} site orders of test_shift): the same gates, three audit rules\n\n"
             f"{streams_table(held)}",
             f"### Calibration-site streams ({seeds} re-drawn cal / test_in splits): choosing floor and margin\n\n"
             f"{streams_table(cal)}\n"
             f"Pick: **{chosen or 'none (no candidate matched the default)'}**"]
    if chosen:
        k = float(chosen.split("κ ")[1].split(")")[0])
        evals = {chosen: cands[chosen], f"alearned (κ {k:g}), uniform": (alearned(k), flat(rate)),
                 REFERENCE: (alearned(0.2), flat(rate))}
        parts.append(f"### Held-out streams: the pick\n\n{streams_table(audit_streams(rows, eps, evals, 'held_out', rate, seeds))}")
    return "\n\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", type=Path, required=True)
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--audit-rate", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    jobs = [(a.collect, eps, a.audit_rate, a.seeds) for eps in (0.05, 0.1)]
    with ProcessPoolExecutor(a.workers) as ex:
        sections = list(ex.map(run_eps, jobs))
    head = [
        f"# Uniform vs risk-targeted audits: {a.collect.stem}",
        f"- collection: `{a.collect.as_posix()}`; probabilities debiased+T (M1); audit budget {a.audit_rate:g} of acted "
        "steps; escalated steps are always labelled\n"
        "- targeted: q ∝ √(ê(1 − ê)) (alearned, doubly-robust feedback) or √ê (arisk, importance weighting), ê the "
        "cross-fitted error model; scaled so the mean q over the calibration steps the gate acts on is the budget; "
        "clipped to [floor, 1] (`surety.gate.audit.AuditPolicy`)\n"
        "- every arm sees the same streams and audit coins (an acted step is audited iff u < q); 'uniform at x' "
        "spends the targeted arm's realised audit rate, for an equal-cost comparison\n"
        "- floor and margin were chosen on calibration-site streams only (re-drawn cal / test_in splits, test_in "
        "streamed in blocks of sites); held-out numbers were not used to choose\n"
        "- supervisor calls/step = escalations + audits per step; tail = the last 200 steps",
    ]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n\n".join(head + sections) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
