import numpy as np

_PHI = (np.sqrt(5) - 1) / 2


class TemperatureScaler:
    def __init__(self):
        self.T = 1.0

    @staticmethod
    def _rescale(p: np.ndarray, t: float) -> np.ndarray:
        q = np.clip(p, 1e-6, None) ** (1 / t)
        return q / q.sum(axis=-1, keepdims=True)

    def fit(self, probs: list[dict[str, float]], gold: list[str],
            grid: tuple[float, float, int] = (0.3, 5.0, 60)) -> float:
        ids = sorted({k for p in probs for k in p})
        P = np.array([[p.get(k, 0.0) for k in ids] for p in probs])
        y = np.array([ids.index(g) for g in gold])

        def nll(t: float) -> float:
            q = self._rescale(P, t)
            return float(-np.log(np.clip(q[np.arange(len(y)), y], 1e-12, None)).mean())

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
