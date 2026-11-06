import math
from typing import Protocol

import numpy as np

from surety.types import GateDecision

_GRID = np.round(np.arange(0, 1.0001, 0.01), 2)
MIN_GROUP = 30


class GatePolicy(Protocol):
    name: str

    def fit(self, probs: list[dict[str, float]], gold: list[str]) -> None: ...

    def decide(self, probs: dict[str, float]) -> GateDecision: ...

    def update(self, probs: dict[str, float], gold: str, weight: float = 1.0) -> None: ...


def _act(option_id: str, reason: str) -> GateDecision:
    return GateDecision("act", option_id, [option_id], reason)


def _escalate(candidates: list[str], reason: str) -> GateDecision:
    return GateDecision("escalate", None, candidates, reason)


def _top(probs: dict[str, float]) -> tuple[str, float]:
    option_id = max(probs, key=probs.get)
    return option_id, probs[option_id]


def _threshold(probs: dict[str, float], tau: float) -> GateDecision:
    option_id, p = _top(probs)
    return _act(option_id, "confident") if p >= tau else _escalate([], "low_confidence")


def _fit_tau(probs: list[dict[str, float]], gold: list[str], eps: float) -> float:
    tops = [_top(p) for p in probs]
    p_max = np.array([p for _, p in tops])
    wrong = np.array([o != g for (o, _), g in zip(tops, gold)])
    for tau in _GRID:
        if np.mean((p_max >= tau) & wrong) <= eps:
            return float(tau)
    return math.inf


def _quantile(scores: np.ndarray, level: float) -> float:
    return float(np.quantile(scores, level, method="higher"))


class _Static:
    def fit(self, probs: list[dict[str, float]], gold: list[str]) -> None:
        pass

    def update(self, probs: dict[str, float], gold: str, weight: float = 1.0) -> None:
        pass


class FixedThreshold(_Static):
    def __init__(self, tau: float):
        self.tau = tau
        self.name = f"fixed@{tau}"

    def decide(self, probs: dict[str, float]) -> GateDecision:
        return _threshold(probs, self.tau)


class Margin(_Static):
    def __init__(self, delta: float):
        self.delta = delta
        self.name = f"margin@{delta}"

    def decide(self, probs: dict[str, float]) -> GateDecision:
        p1, p2 = (sorted(probs.values(), reverse=True) + [0.0])[:2]
        if p1 - p2 >= self.delta:
            return _act(_top(probs)[0], "margin")
        return _escalate([], "low_margin")


class FittedThreshold(FixedThreshold):
    def __init__(self, eps: float):
        super().__init__(math.inf)
        self.eps = eps
        self.name = f"fitted@{eps}"

    def fit(self, probs: list[dict[str, float]], gold: list[str]) -> None:
        self.tau = _fit_tau(probs, gold, self.eps)


class PerChoiceThreshold(_Static):
    def __init__(self, eps: float):
        self.eps = eps
        self.name = f"per_choice@{eps}"
        self.tau = math.inf
        self.taus: dict[str, float] = {}

    def fit(self, probs: list[dict[str, float]], gold: list[str]) -> None:
        self.tau = _fit_tau(probs, gold, self.eps)
        groups: dict[str, tuple[list, list]] = {}
        for p, g in zip(probs, gold):
            ps, gs = groups.setdefault(_top(p)[0], ([], []))
            ps.append(p)
            gs.append(g)
        self.taus = {o: _fit_tau(ps, gs, self.eps) for o, (ps, gs) in groups.items() if len(ps) >= MIN_GROUP}

    def decide(self, probs: dict[str, float]) -> GateDecision:
        return _threshold(probs, self.taus.get(_top(probs)[0], self.tau))


class SplitConformal(_Static):
    def __init__(self, eps: float):
        self.eps = eps
        self.name = f"conformal@{eps}"
        self.scores = np.array([])
        self.q = math.inf

    def fit(self, probs: list[dict[str, float]], gold: list[str]) -> None:
        self.scores = np.array([1 - p.get(g, 0.0) for p, g in zip(probs, gold)])
        n = len(self.scores)
        k = math.ceil((n + 1) * (1 - self.eps))
        # k > n: too few calibration items for 1-eps coverage, so only the full set is valid
        self.q = _quantile(self.scores, k / n) if 0 < n and k <= n else math.inf

    def _q(self) -> float:
        return self.q

    def prediction_set(self, probs: dict[str, float]) -> list[str]:
        q = self._q()
        return sorted((o for o, p in probs.items() if 1 - p <= q), key=probs.get, reverse=True)

    def decide(self, probs: dict[str, float]) -> GateDecision:
        c = self.prediction_set(probs)
        if len(c) == 1:
            return _act(c[0], "singleton")
        return _escalate(c, "set" if c else "empty_set")


class AdaptiveConformal(SplitConformal):
    def __init__(self, eps: float, gamma: float = 0.02, alpha_min: float = 0.001, alpha_max: float = 0.5):
        super().__init__(eps)
        self.name = f"aci@{eps}"
        self.gamma = gamma
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.alpha = eps

    def _q(self) -> float:
        if self.alpha <= self.alpha_min or not len(self.scores):
            return math.inf
        return _quantile(self.scores, 1 - self.alpha)

    def update(self, probs: dict[str, float], gold: str, weight: float = 1.0) -> None:
        err = float(gold not in self.prediction_set(probs))
        alpha = self.alpha + self.gamma * (self.eps - weight * err)
        self.alpha = min(max(alpha, self.alpha_min), self.alpha_max)
