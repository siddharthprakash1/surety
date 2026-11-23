"""Risk control on a learned error score.

RiskConformal acts when the top probability clears a calibrated bar, so it ranks steps by p_top alone. Here a small
logistic model predicts P(top option wrong) from five features of the probability vector (top probability, margin,
normalised entropy, log option count, top is a terminal option) and the gate acts when that prediction is low.
On M1's held-out sites the model ranks errors better (AUROC 0.84 vs 0.70 for p_top; `results/m1/kev_compact_learned.md`).

Validity: the model is cross-fitted (K folds) on the calibration rows; each cal row is scored by the fold model that
did not see it, the threshold is calibrated on those out-of-fold scores, and new steps are scored by the mean of the
fold models. The loss 1{act ∧ wrong} is non-decreasing in the threshold, so
- delta=None: conformal risk control (Angelopoulos et al. 2022): the largest τ with n/(n+1)·R̂(τ) + 1/(n+1) ≤ ε, so
  E[autonomous error] ≤ ε (approximately: the out-of-fold scores come from models fitted on (K-1)/K of the rows);
- delta given: Learn-then-Test (Angelopoulos et al. 2021), fixed-sequence binomial tests from the most conservative τ
  upward, so P(autonomous error > ε) ≤ δ under the same approximation.
"""
import math

import numpy as np

from surety.gate.policies import SplitConformal, _OnlineAlpha, _conformal_q, _escalate, _lac_set, _top
from surety.types import GateDecision

TERMINAL = ("none_of_these", "DONE", "BLOCKED")
FEATURES = ("p_top", "margin", "entropy", "log_options", "top_terminal")


def features(probs: dict[str, float], terminal: tuple[str, ...] = TERMINAL) -> list[float]:
    ps = sorted(probs.values(), reverse=True) + [0.0]
    n = len(probs)
    entropy = -sum(p * math.log(p) for p in probs.values() if p > 0) / math.log(n) if n > 1 else 0.0
    return [ps[0], ps[0] - ps[1], entropy, math.log(max(n, 1)), float(_top(probs)[0] in terminal)]


class ErrorModel:
    """L2-regularised logistic regression (Newton steps) on standardised features."""

    def __init__(self, l2: float = 1.0, iters: int = 30):
        self.l2, self.iters = l2, iters
        self.mu = self.sd = self.w = np.zeros(0)

    def _design(self, x: np.ndarray) -> np.ndarray:
        return np.c_[np.ones(len(x)), (x - self.mu) / self.sd]

    def fit(self, x: np.ndarray, y: np.ndarray) -> "ErrorModel":
        self.mu, self.sd = x.mean(0), x.std(0) + 1e-9
        z = self._design(x)
        reg = self.l2 * np.diag(np.r_[0.0, np.ones(z.shape[1] - 1)])
        self.w = np.zeros(z.shape[1])
        for _ in range(self.iters):
            p = 1 / (1 + np.exp(-z @ self.w))
            step = np.linalg.solve(z.T @ (z * (p * (1 - p))[:, None]) + reg, z.T @ (p - y) + reg @ self.w)
            self.w -= step
            if np.abs(step).max() < 1e-8:
                break
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-self._design(x) @ self.w))


def _boundaries(scores: np.ndarray, wrong: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Thresholds at which the acted set changes (distinct sorted scores) and the wrong-and-acted count at each."""
    order = np.argsort(scores, kind="mergesort")
    s, cum = scores[order], np.cumsum(wrong[order])
    last = np.r_[s[1:] != s[:-1], True]  # last index of each run of equal scores
    return s[last], cum[last]


def crc_threshold(scores: np.ndarray, wrong: np.ndarray, eps: float) -> float:
    """Largest τ (act iff score ≤ τ) with n/(n+1)·R̂(τ) + 1/(n+1) ≤ ε; -inf if none."""
    n = len(scores)
    if not n:
        return -math.inf
    tau, k = _boundaries(scores, wrong)
    ok = np.nonzero(k / (n + 1) + 1 / (n + 1) <= eps)[0]
    return float(tau[ok[-1]]) if len(ok) else -math.inf


def binom_cdf(k: int, n: int, p: float) -> float:
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    i = np.arange(k + 1)
    logs = (math.lgamma(n + 1) - np.array([math.lgamma(j + 1) + math.lgamma(n - j + 1) for j in i])
            + i * math.log(p) + (n - i) * math.log1p(-p))
    top = logs.max()
    return float(min(1.0, math.exp(top) * np.exp(logs - top).sum()))


def ltt_threshold(scores: np.ndarray, wrong: np.ndarray, eps: float, delta: float) -> float:
    """Learn-then-Test: walk τ upward from the most conservative, rejecting H_τ: R(τ) > ε while the binomial
    p-value P(Bin(n, ε) ≤ n·R̂(τ)) ≤ δ; stop at the first failure (fixed-sequence testing). -inf if none."""
    n = len(scores)
    tau, k = _boundaries(scores, wrong)
    best = -math.inf
    for t, kk in zip(tau, k):
        if binom_cdf(int(kk), n, eps) > delta:
            break
        best = float(t)
    return best


class LearnedRisk(SplitConformal):
    """Act on the top option iff the cross-fitted error model's prediction is ≤ τ (CRC, or LTT when delta is set)."""

    def __init__(self, eps: float, delta: float | None = None, folds: int = 5, l2: float = 1.0, seed: int = 0,
                 terminal: tuple[str, ...] = TERMINAL):
        super().__init__(eps)
        self.delta, self.folds, self.l2, self.seed, self.terminal = delta, folds, l2, seed, terminal
        self.name = f"learned@{eps}" if delta is None else f"learned_pac@{eps}"
        self.models: list[ErrorModel] = []
        self.oof = self.wrong = np.array([])
        self.tau = -math.inf
        self.lac_q = math.inf

    def _x(self, probs: list[dict[str, float]]) -> np.ndarray:
        return np.array([features(p, self.terminal) for p in probs], float).reshape(len(probs), len(FEATURES))

    def fit(self, probs: list[dict[str, float]], gold: list[str]) -> None:
        self.lac_q = _conformal_q(np.array([1 - p.get(g, 0.0) for p, g in zip(probs, gold)]), self.eps)
        n = len(probs)
        x = self._x(probs)
        self.wrong = np.array([_top(p)[0] != g for p, g in zip(probs, gold)], float)
        k = min(self.folds, n)
        if k < 2 or self.wrong.min(initial=1) == self.wrong.max(initial=0):
            self.models, self.oof, self.tau = [], np.array([]), -math.inf  # too few rows or one class: escalate all
            return
        parts = np.array_split(np.random.default_rng(self.seed).permutation(n), k)
        self.oof = np.empty(n)
        self.models = []
        for j, held in enumerate(parts):
            train = np.concatenate([q for i, q in enumerate(parts) if i != j])
            model = ErrorModel(self.l2).fit(x[train], self.wrong[train])
            self.oof[held] = model.predict(x[held])
            self.models.append(model)
        self.tau = self._threshold(self.eps)

    def _threshold(self, level: float) -> float:
        if self.delta is None:
            return crc_threshold(self.oof, self.wrong, level)
        return ltt_threshold(self.oof, self.wrong, level, self.delta)

    def score(self, probs: dict[str, float]) -> float:
        """Predicted probability that the top option is wrong (mean of the fold models); inf before fitting."""
        if not self.models:
            return math.inf
        x = self._x([probs])
        return float(np.mean([m.predict(x)[0] for m in self.models]))

    def _tau(self) -> float:
        return self.tau

    def prediction_set(self, probs: dict[str, float]) -> list[str]:
        if self.score(probs) <= self._tau():
            return [_top(probs)[0]]
        lac = _lac_set(probs, self.lac_q)
        return lac if len(lac) >= 2 else []

    def decide(self, probs: dict[str, float]) -> GateDecision:
        c = self.prediction_set(probs)
        if len(c) == 1:
            return GateDecision("act", c[0], c, "learned")
        # as RiskConformal: the LAC set when it has ≥ 2 options, else the full list
        return _escalate(c, "set") if c else _escalate([], "full_view")


class AdaptiveLearnedRisk(_OnlineAlpha, LearnedRisk):
    """LearnedRisk with its CRC level α updated online from weighted feedback (ACI, as AdaptiveRisk), plus:

    - feedback="dr": an acted step's loss estimate is ê + w·(ℓ − ê), with ê the model's predicted error: an unaudited
      step (w = 0) counts ê instead of 0, an audited one (w = 1/r) corrects it. Unbiased for audits at a known rate, as
      the plain estimate w·ℓ ("ipw"), with lower variance when ê tracks the error; escalated steps have loss 0.
    - margin κ: the online target is (1 − κ)·ε, a buffer for finite streams (ACI controls the long-run average only).
    gamma 0.005, κ 0.2 and "dr" were chosen on M1's held-out streams (results/m1/kev_compact_learned.md).
    """

    def __init__(self, eps: float, gamma: float = 0.005, alpha_min: float = 0.001, alpha_max: float = 0.5,
                 margin: float = 0.2, feedback: str = "dr", **kw):
        super().__init__(eps, **kw)
        if feedback not in ("dr", "ipw") or not 0 <= margin < 1:
            raise ValueError(f"feedback {feedback!r} (dr|ipw), margin {margin} (in [0, 1))")
        self.name = f"alearned@{eps}"
        self.gamma, self.alpha_min, self.alpha_max = gamma, alpha_min, alpha_max
        self.margin, self.feedback = margin, feedback
        self.target = (1 - margin) * eps
        self.alpha_raw = self.target
        self._cache: tuple[float, float] | None = None

    def _tau(self) -> float:
        if self.alpha <= self.alpha_min or not self.models:
            return -math.inf
        if self._cache is None or self._cache[0] != self.alpha:
            self._cache = (self.alpha, self._threshold(self.alpha))
        return self._cache[1]

    def loss_estimate(self, probs: dict[str, float], gold: str, weight: float) -> float:
        c = self.prediction_set(probs)
        if len(c) != 1:
            return 0.0
        loss = float(gold not in c)
        if self.feedback == "ipw":
            return weight * loss
        e = self.score(probs)
        return e + weight * (loss - e)

    def update(self, probs: dict[str, float], gold: str, weight: float = 1.0) -> None:
        est = self.loss_estimate(probs, gold, weight)
        self.alpha_raw = min(self.alpha_raw + self.gamma * (self.target - est), self.alpha_max)
