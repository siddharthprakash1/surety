import math
import re
from collections import Counter

import numpy as np

_PHI = (np.sqrt(5) - 1) / 2
# page elements: WebJev E12 / clickable_12, BrowserGym bid:12, Playwright e12 (ids recur across pages, name different things)
ELEMENT_ID = r"^(?:E\d+|e\d+|bid:.+|clickable(?:_\d+)?)$"


class PriorCorrector:
    """One log-prior offset per option id that recurs across steps (WebJev's action vocabulary: CLICK, TYPE_TEXT,
    SCROLL_DOWN, ..., DONE, BLOCKED): p'(o) ∝ p(o)·exp(b_o). A zero-shot executor can carry a strong label bias, which
    one offset per id removes: Kev-4B ranks BLOCKED first on 30 % of WebJev action-prediction steps and is wrong on
    99 % of them. b is fitted by L2-penalised maximum likelihood of the gold option on calibration steps (Newton);
    page-element ids and ids seen in fewer than min_rows steps keep b = 0. Contextual calibration (Zhao et al. 2021),
    with the bias estimated from labelled steps instead of a content-free input."""

    def __init__(self, min_rows: int = 30, l2: float = 1.0, iters: int = 50, exclude: str = ELEMENT_ID):
        self.min_rows, self.l2, self.iters, self.exclude = min_rows, l2, iters, re.compile(exclude)
        self.b: dict[str, float] = {}

    def fit(self, probs: list[dict[str, float]], gold: list[str]) -> dict[str, float]:
        counts = Counter(o for p in probs for o in p)
        vocab = sorted(o for o, c in counts.items() if c >= self.min_rows and not self.exclude.match(o))
        index = {o: k for k, o in enumerate(vocab)}
        rows = [(np.log(np.clip(np.array(list(p.values())), 1e-12, None)),
                 np.array([index.get(o, -1) for o in p]), list(p).index(g))
                for p, g in zip(probs, gold) if g in p and any(o in index for o in p)]
        def shifted(b: np.ndarray, lg: np.ndarray, ids: np.ndarray) -> np.ndarray:
            return lg + np.where(ids >= 0, b[np.maximum(ids, 0)], 0.0)

        def objective(b: np.ndarray) -> float:
            ll = 0.0
            for lg, ids, y in rows:
                z = shifted(b, lg, ids)
                ll += z[y] - (z.max() + math.log(np.exp(z - z.max()).sum()))
            return ll - self.l2 / 2 * float(b @ b)

        b = np.zeros(len(vocab))
        f = objective(b) if vocab else 0.0
        for _ in range(self.iters if vocab else 0):
            grad, hess = -self.l2 * b, -self.l2 * np.eye(len(vocab))
            for lg, ids, y in rows:
                z = shifted(b, lg, ids)
                q = np.exp(z - z.max())
                q /= q.sum()
                mass = np.zeros(len(vocab))
                inv = ids >= 0
                np.add.at(mass, ids[inv], q[inv])  # probability mass on each vocabulary id
                grad -= mass
                if ids[y] >= 0:
                    grad[ids[y]] += 1.0
                hess -= np.diag(mass) - np.outer(mass, mass)
            step = -np.linalg.solve(hess, grad)  # Newton ascent direction
            t = 1.0
            while t > 1e-6 and (new := objective(b + t * step)) < f:  # backtrack: plain Newton can overshoot here
                t /= 2
            if t <= 1e-6:
                break
            b, f = b + t * step, new
            if np.abs(t * step).max() < 1e-8:
                break
        self.b = {o: float(v) for o, v in zip(vocab, b)}
        return self.b

    def apply(self, probs: dict[str, float]) -> dict[str, float]:
        if not self.b:
            return dict(probs)
        z = {o: math.log(max(p, 1e-12)) + self.b.get(o, 0.0) for o, p in probs.items()}
        m = max(z.values())
        e = {o: math.exp(v - m) for o, v in z.items()}
        s = sum(e.values())
        return {o: v / s for o, v in e.items()}


class TemperatureScaler:
    def __init__(self):
        self.T = 1.0

    @staticmethod
    def _rescale(p: np.ndarray, t: float) -> np.ndarray:
        q = np.clip(p, 1e-6, None) ** (1 / t)
        return q / q.sum(axis=-1, keepdims=True)

    def fit(self, probs: list[dict[str, float]], gold: list[str],
            grid: tuple[float, float, int] = (0.3, 5.0, 60)) -> float:
        logs = [np.log(np.clip(np.array(list(p.values())), 1e-6, None)) for p in probs]
        gi = [list(p).index(g) for p, g in zip(probs, gold)]

        def nll(t: float) -> float:
            return float(np.mean([-(lg[y] / t) + np.logaddexp.reduce(lg / t)
                                  for lg, y in zip(logs, gi)]))

        ts = np.linspace(*grid)
        i = int(np.argmin([nll(t) for t in ts]))
        lo, hi = ts[max(i - 1, 0)], ts[min(i + 1, len(ts) - 1)]
        a, b = hi - _PHI * (hi - lo), lo + _PHI * (hi - lo)
        for _ in range(60):
            if nll(a) < nll(b):
                hi = b
            else:
                lo = a
            a, b = hi - _PHI * (hi - lo), lo + _PHI * (hi - lo)
        self.T = float((lo + hi) / 2)
        return self.T

    def apply(self, probs: dict[str, float]) -> dict[str, float]:
        ids = list(probs)
        q = self._rescale(np.array([probs[k] for k in ids]), self.T)
        return {k: float(v) for k, v in zip(ids, q)}
