import math
import warnings

import pytest

from surety.calib import metrics as m

# Every public metric that divides by the row count, called on empty input.
EMPTY_METRIC_CALLS = [
    ("accuracy", lambda: [m.accuracy([], [])]),
    ("ece", lambda: [m.ece([], [])]),
    ("brier", lambda: [m.brier([], [])]),
    ("confident_error_rate", lambda: [m.confident_error_rate([], [])]),
    ("coverage", lambda: [m.coverage([], [])]),
    ("mean_set_size", lambda: [m.mean_set_size([])]),
    ("automation_rate", lambda: [m.automation_rate([])]),
    ("autonomous_error_rate", lambda: [m.autonomous_error_rate([], [])]),
    ("risk_coverage_curve", lambda: [
        r[k] for r in m.risk_coverage_curve([], [], [0.5, 0.9]) for k in ("acted", "risk", "error_rate_all")
    ]),
]


@pytest.mark.parametrize("call", [c for _, c in EMPTY_METRIC_CALLS], ids=[n for n, _ in EMPTY_METRIC_CALLS])
def test_empty_input_is_nan_without_runtime_warning(call):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        values = call()
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert values and all(math.isnan(v) for v in values)


def test_ece_perfectly_calibrated_is_zero():
    probs = [{"a": 0.5, "b": 0.5}, {"a": 0.5, "b": 0.5}]
    assert m.ece(probs, ["a", "b"]) == pytest.approx(0.0)


def test_ece_overconfident():
    probs = [{"a": 0.9, "b": 0.1}] * 2
    assert m.ece(probs, ["a", "b"]) == pytest.approx(0.4)


def test_ece_confidence_one_lands_in_last_bin():
    # p_max == 1.0 maps to bin 15 before the clamp; it must count in the last bin, not be dropped.
    probs = [{"a": 0.0, "b": 1.0}]
    assert m.ece(probs, ["a"]) == pytest.approx(1.0)


def test_ece_sums_over_several_nonempty_bins():
    # bin 10 (conf 0.7): 4 rows, acc 3/4 -> |0.75 - 0.7| = 0.05, weight 4/8
    # bin 13 (conf 0.9): 2 rows, acc 1/2 -> |0.50 - 0.9| = 0.40, weight 2/8
    # bin 8  (conf 0.55): 2 rows, acc 2/2 -> |1.00 - 0.55| = 0.45, weight 2/8
    # ECE = 0.5*0.05 + 0.25*0.40 + 0.25*0.45 = 0.2375
    probs = [{"a": 0.7, "b": 0.3}] * 4 + [{"a": 0.9, "b": 0.1}] * 2 + [{"a": 0.55, "b": 0.45}] * 2
    gold = ["a", "a", "a", "b", "a", "b", "a", "a"]
    assert m.ece(probs, gold) == pytest.approx(0.2375)


def test_accuracy():
    probs = [{"a": 0.6, "b": 0.4}, {"a": 0.6, "b": 0.4}]
    assert m.accuracy(probs, ["a", "b"]) == 0.5


def test_brier_one_hot_correct_is_zero():
    assert m.brier([{"a": 1.0, "b": 0.0}], ["a"]) == 0.0


def test_brier_uniform_two_options():
    assert m.brier([{"a": 0.5, "b": 0.5}], ["a"]) == pytest.approx(0.5)


def test_confident_error_rate():
    probs = [{"a": 0.95, "b": 0.05}, {"a": 0.95, "b": 0.05}, {"a": 0.6, "b": 0.4}]
    assert m.confident_error_rate(probs, ["b", "a", "b"]) == pytest.approx(1 / 3)


def test_coverage_and_set_size():
    sets = [["a"], ["a", "b"], ["c"]]
    assert m.coverage(sets, ["a", "b", "a"]) == pytest.approx(2 / 3)
    assert m.mean_set_size(sets) == pytest.approx(4 / 3)


def test_automation_rate_counts_only_singletons():
    sets = [["a"], ["a", "b"], ["c"]]
    assert m.automation_rate(sets) == pytest.approx(2 / 3)


def test_autonomous_error_counts_only_singletons():
    sets = [["a"], ["a", "b"], ["c"]]
    assert m.autonomous_error_rate(sets, ["b", "b", "c"]) == pytest.approx(1 / 3)


def test_risk_coverage_monotone_acted():
    probs = [{"a": 0.9, "b": 0.1}, {"a": 0.7, "b": 0.3}, {"a": 0.55, "b": 0.45}]
    rc = m.risk_coverage_curve(probs, ["a", "b", "b"], [0.0, 0.6, 0.8, 0.95])
    acted = [r["acted"] for r in rc]
    assert acted == sorted(acted, reverse=True)
    assert rc[0]["risk"] == pytest.approx(2 / 3)
    assert rc[1]["risk"] == pytest.approx(0.5)
    assert rc[1]["error_rate_all"] == pytest.approx(1 / 3)
    assert rc[3] == {"tau": 0.95, "acted": 0.0, "risk": 0.0, "error_rate_all": 0.0}


def test_wilson_interval():
    assert m.wilson(10, 100) == pytest.approx((0.0552, 0.1744), abs=1e-4)
    lo, hi = m.wilson(0, 600)
    assert lo == 0.0 and hi == pytest.approx(0.00636, abs=1e-5)
