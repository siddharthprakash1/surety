"""How likely a step the gate acted on is to be audited. Audits are the adaptive gates' only error signal on acted
steps, and their noise sets the margin a finite stream needs (docs/research/2026-11-24-improvements.md §3): with audits
at rate r the estimated count of acted errors varies by about √(εT(1 − r)/r).

Targeted audits keep the budget but spend it where errors are likely. A step is audited with probability
q ∝ √v(ê), the allocation that minimises the variance of the weighted error estimate for a fixed expected number of
audits (Neyman allocation). ê is a cross-fitted estimate of P(top option wrong) (gate.learned), and v(ê) = ê(1 − ê)
for doubly-robust feedback (the residual's variance) or ê for plain importance weighting. The scale is set at fit time
so that the mean of q over the calibration steps the gate acts on equals r; q is clipped to [floor, 1], so every acted
step can be audited and no weight exceeds 1/floor. Weights 1/q keep the feedback unbiased, as with uniform audits: q is
fixed before the coin is flipped.
"""
import math

import numpy as np

from surety.gate.learned import LearnedRisk


def _scale(v: np.ndarray, rate: float, floor: float) -> float | None:
    """s with mean(clip(s·v, floor, 1)) = rate; None when no s reaches it (e.g. v all 0)."""
    def mean_q(s: float) -> float:
        return float(np.clip(s * v, floor, 1.0).mean())

    if not len(v):
        return None
    if mean_q(0.0) >= rate:
        return 0.0
    hi = 1.0
    while mean_q(hi) < rate:
        hi *= 2
        if hi > 1e12:
            return None
    lo = 0.0
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if mean_q(mid) < rate else (lo, mid)
    return hi


class AuditPolicy:
    """Audit probability of an acted step: `rate` for every step (uniform), or risk-targeted at the same mean rate."""

    def __init__(self, rate: float, targeted: bool = False, floor: float = 0.05, feedback: str = "dr", seed: int = 0):
        if not (0 < rate <= 1 and 0 < floor <= rate) or feedback not in ("dr", "ipw"):
            raise ValueError(f"rate {rate} in (0, 1], floor {floor} in (0, rate], feedback {feedback!r} (dr|ipw)")
        self.rate, self.targeted, self.floor, self.feedback = rate, targeted, floor, feedback
        self.scorer = LearnedRisk(rate, seed=seed)   # only its cross-fitted error model is used
        self.scale: float | None = None

    @property
    def name(self) -> str:
        return f"targeted (floor {self.floor:g}, {self.feedback})" if self.targeted else "uniform"

    def _v(self, e: float) -> float:
        e = min(max(e, 0.0), 1.0)
        return math.sqrt(e * (1 - e)) if self.feedback == "dr" else math.sqrt(e)

    def fit(self, probs: list[dict[str, float]], gold: list[str], acted: list[dict[str, float]]) -> None:
        """Fit the error model on the calibration steps; `acted`: those of them the gate acts on, whose mean audit
        probability is set to `rate`. Falls back to uniform when the model or the scale cannot be fitted."""
        self.scale = None
        if not self.targeted:
            return
        self.scorer.fit(probs, gold)
        if self.scorer.models:
            self.scale = _scale(np.array([self._v(self.scorer.score(p)) for p in acted]), self.rate, self.floor)

    def prob(self, probs: dict[str, float]) -> float:
        if self.scale is None:
            return self.rate
        return min(1.0, max(self.floor, self.scale * self._v(self.scorer.score(probs))))
