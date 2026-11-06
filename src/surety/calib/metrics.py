import numpy as np


def _top(probs, gold):
    conf = np.array([max(p.values()) for p in probs])
    wrong = np.array([max(p, key=p.get) != g for p, g in zip(probs, gold)])
    return conf, wrong


def accuracy(probs, gold) -> float:
    return float(1.0 - _top(probs, gold)[1].mean())


def ece(probs, gold, n_bins: int = 15) -> float:
    conf, wrong = _top(probs, gold)
    correct = ~wrong
    idx = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(total)


def brier(probs, gold) -> float:
    return float(np.mean([sum((v - (o == g)) ** 2 for o, v in p.items()) for p, g in zip(probs, gold)]))


def confident_error_rate(probs, gold, threshold: float = 0.9) -> float:
    conf, wrong = _top(probs, gold)
    return float(np.mean((conf >= threshold) & wrong))


def coverage(sets, gold) -> float:
    return float(np.mean([g in s for s, g in zip(sets, gold)]))


def mean_set_size(sets) -> float:
    return float(np.mean([len(s) for s in sets]))


def automation_rate(sets) -> float:
    return float(np.mean([len(s) == 1 for s in sets]))


def autonomous_error_rate(sets, gold) -> float:
    return float(np.mean([len(s) == 1 and s[0] != g for s, g in zip(sets, gold)]))


def risk_coverage_curve(probs, gold, thresholds) -> list[dict]:
    conf, wrong = _top(probs, gold)
    out = []
    for tau in thresholds:
        act = conf >= tau
        out.append({
            "tau": tau,
            "acted": float(act.mean()),
            "risk": float(wrong[act].mean()) if act.any() else 0.0,
            "error_rate_all": float((act & wrong).mean()),
        })
    return out
