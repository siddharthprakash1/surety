import pytest

from surety.calib import metrics as m


def test_ece_perfectly_calibrated_is_zero():
    probs = [{"a": 0.5, "b": 0.5}, {"a": 0.5, "b": 0.5}]
    assert m.ece(probs, ["a", "b"]) == pytest.approx(0.0)


def test_ece_overconfident():
    probs = [{"a": 0.9, "b": 0.1}] * 2
    assert m.ece(probs, ["a", "b"]) == pytest.approx(0.4)


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


def test_autonomous_error_counts_only_singletons():
    sets = [["a"], ["a", "b"], ["c"]]
    assert m.automation_rate(sets) == pytest.approx(2 / 3)
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
