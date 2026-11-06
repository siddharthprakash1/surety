import numpy as np
import pytest

from surety.gate import (
    AdaptiveConformal,
    FittedThreshold,
    FixedThreshold,
    Margin,
    PerChoiceThreshold,
    SplitConformal,
)

TWO = {"a": 0.9, "b": 0.1}
CAL_TWO = ([TWO] * 20, ["a"] * 20)


def synthetic(n, rng, k=5):
    p = np.exp(rng.normal(size=(n, k)) * 2)
    p /= p.sum(axis=1, keepdims=True)
    gold = [f"o{rng.choice(k, p=row)}" for row in p]
    return [{f"o{j}": float(v) for j, v in enumerate(row)} for row in p], gold


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    return synthetic(2000, rng), synthetic(2000, rng)


def view(d):
    return d.action, d.option_id, d.candidate_set


def autonomous_error(policy, probs, gold):
    ds = [policy.decide(p) for p in probs]
    return np.mean([d.action == "act" and d.option_id != y for d, y in zip(ds, gold)])


def test_fixed_threshold_acts_above_tau():
    g = FixedThreshold(0.7)
    assert view(g.decide({"a": 0.8, "b": 0.2})) == ("act", "a", ["a"])
    assert view(g.decide({"a": 0.6, "b": 0.4})) == ("escalate", None, [])


def test_margin_escalates_close_top_two():
    g = Margin(0.1)
    assert view(g.decide({"a": 0.45, "b": 0.4, "c": 0.15})) == ("escalate", None, [])
    assert view(g.decide({"a": 0.6, "b": 0.3, "c": 0.1})) == ("act", "a", ["a"])


def test_fitted_threshold_meets_eps_on_calibration(data):
    (probs, gold), _ = data
    g = FittedThreshold(0.1)
    g.fit(probs, gold)
    looser = FixedThreshold(round(g.tau - 0.01, 2))
    assert autonomous_error(g, probs, gold) <= 0.1 < autonomous_error(looser, probs, gold)


def test_per_choice_threshold_unseen_id_uses_global(data):
    (probs, gold), _ = data
    g, f = PerChoiceThreshold(0.1), FittedThreshold(0.1)
    g.fit(probs, gold)
    f.fit(probs, gold)
    assert set(g.taus) == {f"o{j}" for j in range(5)}
    rest = (1 - f.tau) / 2
    assert g.decide({"new": f.tau, "x": rest, "y": rest}).action == "act"
    assert g.decide({"new": f.tau - 0.01, "x": rest, "y": rest + 0.01}).action == "escalate"


def test_split_conformal_coverage_on_synthetic(data):
    (cal_p, cal_g), (test_p, test_g) = data
    g = SplitConformal(0.1)
    g.fit(cal_p, cal_g)
    coverage = np.mean([y in g.decide(p).candidate_set for p, y in zip(test_p, test_g)])
    assert 0.88 <= coverage <= 0.93


def test_conformal_single_option_acts(data):
    (probs, gold), _ = data
    policies = [
        FixedThreshold(0.7),
        Margin(0.1),
        FittedThreshold(0.1),
        PerChoiceThreshold(0.1),
        SplitConformal(0.1),
        AdaptiveConformal(0.1),
    ]
    for g in policies:
        g.fit(probs, gold)
        assert view(g.decide({"a": 1.0})) == ("act", "a", ["a"]), g.name


def test_conformal_set_sorted_by_prob():
    g = SplitConformal(0.1)
    g.fit([{"a": 0.3, "b": 0.7}] * 10, ["a"] * 10)
    assert view(g.decide({"x": 0.2, "y": 0.5, "z": 0.3})) == ("escalate", None, ["y", "z"])


def test_conformal_empty_set_escalates_full_view():
    g = SplitConformal(0.1)
    g.fit(*CAL_TWO)
    assert view(g.decide({"a": 0.5, "b": 0.5})) == ("escalate", None, [])


def test_adaptive_alpha_moves_toward_target():
    g = AdaptiveConformal(0.1)
    g.fit(*CAL_TWO)
    for _ in range(50):
        g.update(TWO, "b")
    low = g.alpha
    assert low < 0.1
    for _ in range(50):
        g.update(TWO, "a")
    assert g.alpha > low


def test_adaptive_weighted_update_scales_by_weight():
    alphas = []
    for w in (1.0, 3.0):
        g = AdaptiveConformal(0.1)
        g.fit(*CAL_TWO)
        g.update(TWO, "b", weight=w)
        alphas.append(g.alpha)
    assert alphas == pytest.approx([0.1 + 0.02 * (0.1 - 1), 0.1 + 0.02 * (0.1 - 3)])


def test_adaptive_alpha_floor_escalates_full_set():
    g = AdaptiveConformal(0.1)
    g.fit(*CAL_TWO)
    g.update(TWO, "b", weight=100)
    assert g.alpha == 0.001
    assert view(g.decide({"b": 0.1, "a": 0.9})) == ("escalate", None, ["a", "b"])
