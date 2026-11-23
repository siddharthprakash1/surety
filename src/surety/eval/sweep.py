"""Target ε vs realized autonomous error and automation, over calibration draws (as resample.draws)."""
from pathlib import Path

import numpy as np

from surety.calib import metrics as m
from surety.eval.analyze import _sets, _split, _table, variants
from surety.eval.resample import resplit
from surety.gate import FittedThreshold, PerChoiceThreshold, RiskConformal

POLICIES = (FittedThreshold, PerChoiceThreshold, RiskConformal)
GRID = tuple(round(x, 2) for x in np.arange(0.02, 0.201, 0.02))


def eps_sweep(rows: list[dict], eps_grid=GRID, n_seeds: int = 100, split: str = "test_shift") -> list[dict]:
    out = []
    for seed in range(n_seeds):
        rs = resplit(rows, seed)
        probs = variants(rs)["debiased+T"]
        cal, (p, g) = _split(rs, probs, "cal"), _split(rs, probs, split)
        for eps in eps_grid:
            for cls in POLICIES:
                policy = cls(eps)
                policy.fit(*cal)
                sets = _sets(policy, p)
                out.append({"seed": seed, "eps": eps, "policy": policy.name.split("@")[0],
                            "auto_err": m.autonomous_error_rate(sets, g), "automation": m.automation_rate(sets)})
    return out


def _summary(records: list[dict]) -> dict[tuple, dict]:
    out = {}
    for key in sorted({(r["policy"], r["eps"]) for r in records}):
        err = np.array([r["auto_err"] for r in records if (r["policy"], r["eps"]) == key])
        auto = np.array([r["automation"] for r in records if (r["policy"], r["eps"]) == key])
        out[key] = {"err": err.mean(), "lo": np.percentile(err, 2.5), "hi": np.percentile(err, 97.5),
                    "over": float(np.mean(err > key[1])), "auto": auto.mean()}
    return out


def sweep_table(records: list[dict]) -> str:
    body = [[f"{eps:g}", pol, float(s["err"]), f"[{s['lo']:.3f}, {s['hi']:.3f}]", s["over"], float(s["auto"])]
            for (pol, eps), s in sorted(_summary(records).items(), key=lambda kv: (kv[0][1], kv[0][0]))]
    return _table(("eps", "policy", "auto_err", "auto_err_2.5–97.5%", "P(auto_err>eps)", "automation"), body)


def sweep_plot(records: list[dict], out_png: Path, title: str = "held-out sites (test_shift)") -> None:
    from matplotlib.figure import Figure

    s = _summary(records)
    fig = Figure(figsize=(9, 3.6))
    ax_err, ax_auto = fig.subplots(1, 2)
    colors = {"fitted": "#eb6834", "per_choice": "#8a63d2", "risk": "#2a78d6"}
    for pol, color in colors.items():
        eps = sorted(e for p, e in s if p == pol)
        if not eps:
            continue
        get = lambda k: [s[(pol, e)][k] for e in eps]  # noqa: E731
        ax_err.plot(eps, get("err"), color=color, lw=2, label=pol)
        ax_err.fill_between(eps, get("lo"), get("hi"), color=color, alpha=0.15, lw=0)
        ax_auto.plot(eps, get("auto"), color=color, lw=2, label=pol)
    top = max(max(e for _, e in s), max(v["hi"] for v in s.values()))
    ax_err.plot([0, top], [0, top], color="#888", lw=1, ls="--", label="target ε")
    ax_err.set(xlabel="target ε", ylabel="autonomous error rate", title=f"Realized error, {title}")
    ax_auto.set(xlabel="target ε", ylabel="automation (fraction acted)", title="Automation")
    for ax in (ax_err, ax_auto):
        ax.grid(alpha=0.3, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    ax_err.legend(frameon=False)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
