"""A live gate from an offline collection: temperature and gate fitted on its cal rows (debiased probs)."""
import math

from surety.decide import TemperatureScaler
from surety.gate import (
    AdaptiveConformal,
    AdaptiveRisk,
    EscalateOnNone,
    FittedThreshold,
    FixedThreshold,
    GatePolicy,
    Margin,
    PerChoiceThreshold,
    RiskConformal,
    SplitConformal,
)

# H4 systems (spec): supervisor-only, executor-only, fixed threshold, Jev-Mobile rule, risk, adaptive risk
POLICIES = {"fixed": FixedThreshold, "margin": Margin, "fitted": FittedThreshold, "per_choice": PerChoiceThreshold,
            "conformal": SplitConformal, "aci": AdaptiveConformal, "risk": RiskConformal, "arisk": AdaptiveRisk}
FITTED = {"fitted", "per_choice", "conformal", "aci", "risk", "arisk"}


def fit_gate(spec: str, rows: list[dict] | None = None, gamma: float = 0.02) -> tuple[GatePolicy, TemperatureScaler | None]:
    """spec: supervisor | executor | escalate_on_none | <policy>:<value>, e.g. fixed:0.7, risk:0.05, arisk:0.05."""
    if spec in ("supervisor", "executor", "escalate_on_none"):
        return {"supervisor": FixedThreshold(math.inf), "executor": FixedThreshold(0.0),
                "escalate_on_none": EscalateOnNone()}[spec], None
    kind, value = spec.split(":")
    policy = POLICIES[kind](float(value), gamma) if kind in ("aci", "arisk") else POLICIES[kind](float(value))
    cal = [r for r in rows or [] if r["split"] == "cal"]
    if kind in FITTED and not cal:
        raise ValueError(f"{kind} needs a collection with cal rows")
    if not cal:
        return policy, None
    scaler = TemperatureScaler()
    gold = [r["gold_id"] for r in cal]
    scaler.fit([r["probs"] for r in cal], gold)
    policy.fit([scaler.apply(r["probs"]) for r in cal], gold)
    return policy, scaler
