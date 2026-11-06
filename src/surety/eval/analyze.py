import json
import random
from pathlib import Path

import numpy as np

from surety.calib import metrics as m
from surety.decide import TemperatureScaler
from surety.gate import (
    AdaptiveConformal,
    AdaptiveRisk,
    FittedThreshold,
    FixedThreshold,
    Margin,
    PerChoiceThreshold,
    RiskConformal,
    SplitConformal,
)

SPLITS = ("cal", "test_in", "test_shift")
TAIL = 200


def load_collection(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # a live collection may still be writing its last line
                if line.endswith("\n"):
                    raise
    return rows


def _split(rows, probs, split):
    idx = [i for i, r in enumerate(rows) if r["split"] == split]
    return [probs[i] for i in idx], [rows[i]["gold_id"] for i in idx]


def _present(rows):
    found = {r["split"] for r in rows}
    return [s for s in SPLITS if s in found]


def variants(rows) -> dict[str, list[dict[str, float]]]:
    debiased = [r["probs"] for r in rows]
    cal = _split(rows, debiased, "cal")
    if not cal[1]:
        raise ValueError("no cal split")
    scaler = TemperatureScaler()
    scaler.fit(*cal)
    return {
        "first_run": [r["runs"][0] for r in rows],
        "debiased": debiased,
        "debiased+T": [scaler.apply(p) for p in debiased],
    }


def _fmt(x):
    return f"{x:.3f}" if isinstance(x, float) else str(x)


def _table(header, body) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(_fmt(c) for c in row) + " |" for row in body]
    return "\n".join(lines) + "\n"


def h1_table(rows, vs=None) -> str:
    vs = vs or variants(rows)
    body = []
    for split in _present(rows):
        for name, probs in vs.items():
            p, g = _split(rows, probs, split)
            body.append([split, name, len(g), m.accuracy(p, g), m.ece(p, g), m.brier(p, g),
                         m.confident_error_rate(p, g)])
    return _table(("split", "variant", "n", "acc", "ECE", "Brier", "conf_err"), body)


def _sets(policy, probs):
    if isinstance(policy, SplitConformal):
        return [policy.prediction_set(p) for p in probs]
    ds = [policy.decide(p) for p in probs]
    return [[d.option_id] if d.action == "act" else list(p) for d, p in zip(ds, probs)]


def h2_table(rows, eps_list=(0.05, 0.1), vs=None) -> str:
    probs = (vs or variants(rows))["debiased+T"]
    cal = _split(rows, probs, "cal")
    tests = [s for s in _present(rows) if s != "cal"]
    body = []
    for eps in eps_list:
        for policy in (FixedThreshold(0.55), FixedThreshold(0.7), Margin(0.1), FittedThreshold(eps),
                       PerChoiceThreshold(eps), SplitConformal(eps), RiskConformal(eps)):
            policy.fit(*cal)
            for split in tests:
                p, g = _split(rows, probs, split)
                sets = _sets(policy, p)
                body.append([str(eps), policy.name, split, len(g), m.autonomous_error_rate(sets, g),
                             m.automation_rate(sets), m.coverage(sets, g), m.mean_set_size(sets)])
    return _table(("eps", "policy", "split", "n", "auto_err", "automation", "coverage", "set_size"), body)


def _stream(policy, probs, gold, r, seed):
    rng = random.Random(seed)
    sets = []
    for x, y in zip(probs, gold):
        c = policy.prediction_set(x)
        sets.append(c)
        if len(c) != 1:
            w = 1.0
        elif rng.random() < r:
            w = 1 / r
        else:
            w = 0.0  # label unobserved: only the ε drift applies
        policy.update(x, y, w)
    return sets


def _aci_row(policy, rate, sets, gold, alpha):
    return [policy.name, rate, len(gold), m.autonomous_error_rate(sets, gold),
            m.autonomous_error_rate(sets[-TAIL:], gold[-TAIL:]), m.automation_rate(sets), alpha]


def aci_simulation(rows, eps, audit_rates=(0.0, 0.05, 0.2), gamma=0.02, seed=0, vs=None) -> str:
    probs = (vs or variants(rows))["debiased+T"]
    cal = _split(rows, probs, "cal")
    sites = sorted({r["site"] for r in rows if r["split"] == "test_shift"})
    random.Random(seed).shuffle(sites)
    block = {s: k for k, s in enumerate(sites)}
    stream = sorted((i for i, r in enumerate(rows) if r["split"] == "test_shift"),
                    key=lambda i: block[rows[i]["site"]])
    p, g = [probs[i] for i in stream], [rows[i]["gold_id"] for i in stream]

    body = []
    for static_cls, adaptive_cls in ((SplitConformal, AdaptiveConformal), (RiskConformal, AdaptiveRisk)):
        static = static_cls(eps)
        static.fit(*cal)
        body.append(_aci_row(static, "-", _sets(static, p), g, "-"))
        for r in audit_rates:
            policy = adaptive_cls(eps, gamma)
            policy.fit(*cal)
            body.append(_aci_row(policy, f"{r:g}", _stream(policy, p, g, r, seed), g, policy.alpha))
    return _table(("policy", "audit_rate", "n", "auto_err", "tail_auto_err", "automation", "final_alpha"), body)


def ablation_table(rows_compact, rows_full) -> str:
    full = {r["id"]: r for r in rows_full}
    pairs = [(r, full[r["id"]]) for r in rows_compact if r["id"] in full]
    body = []
    for rs in zip(*pairs):
        p, g = [r["probs"] for r in rs], [r["gold_id"] for r in rs]
        body.append([", ".join(sorted({r["mode"] for r in rs})), len(rs), m.accuracy(p, g), m.ece(p, g),
                     m.brier(p, g), round(np.mean([r["latency_ms"] for r in rs])),
                     round(np.mean([r["state_chars"] for r in rs])), float(np.mean([r["n_perm"] for r in rs]))])
    return _table(("mode", "n", "acc", "ECE", "Brier", "latency_ms", "state_chars", "n_perm"), body)


def risk_coverage_plot(rows, out_png: Path, vs=None) -> None:
    from matplotlib.figure import Figure

    probs = (vs or variants(rows))["debiased+T"]
    fig = Figure(figsize=(5, 4))
    ax = fig.subplots()
    for split, color in zip(_present(rows), ("#2a78d6", "#eb6834", "#1baf7a")):
        p, g = _split(rows, probs, split)
        # quantile thresholds: overconfident splits pile up above any fixed grid's top step
        taus = sorted({float(t) for t in np.quantile([max(x.values()) for x in p], np.linspace(0, 1, 201))})
        curve = [c for c in m.risk_coverage_curve(p, g, taus) if c["acted"] > 0]
        ax.plot([c["acted"] for c in curve], [c["risk"] for c in curve], color=color, lw=2, label=split)
    ax.set(xlabel="coverage (fraction acted)", ylabel="risk (error among acted)", xlim=(0, 1), ylim=(0, None),
           title="Risk–coverage, debiased+T")
    ax.grid(alpha=0.3, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
