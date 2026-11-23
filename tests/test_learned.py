import math

import numpy as np
import pytest

from surety.gate import RiskConformal
from surety.gate.learned import (
    AdaptiveLearnedRisk,
    ErrorModel,
    LearnedRisk,
    binom_cdf,
    crc_threshold,
    features,
    ltt_threshold,
)


def synthetic(n, rng, k=5, scale=2):
    p = np.exp(rng.normal(size=(n, k)) * scale)
    p /= p.sum(axis=1, keepdims=True)
    gold = [f"o{rng.choice(k, p=row)}" for row in p]
    return [{f"o{j}": float(v) for j, v in enumerate(row)} for row in p], gold


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    return synthetic(2000, rng), synthetic(4000, rng)


def rates(policy, probs, gold):
    ds = [policy.decide(p) for p in probs]
    err = np.mean([d.action == "act" and d.option_id != y for d, y in zip(ds, gold)])
    return err, np.mean([d.action == "act" for d in ds])


def test_features():
    f = features({"a": 0.5, "b": 0.3, "none_of_these": 0.2})
    assert f[:2] == pytest.approx([0.5, 0.2])
    assert f[2] == pytest.approx(-(0.5 * math.log(0.5) + 0.3 * math.log(0.3) + 0.2 * math.log(0.2)) / math.log(3))
    assert f[3:] == pytest.approx([math.log(3), 0.0])
    assert features({"DONE": 0.7, "CLICK": 0.3})[4] == 1.0
    assert features({"a": 1.0}) == pytest.approx([1.0, 1.0, 0.0, 0.0, 0.0])


def test_error_model_ranks_by_signal():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(500, 2))
    y = (rng.random(500) < 1 / (1 + np.exp(-3 * x[:, 0]))).astype(float)
    p = ErrorModel().fit(x, y).predict(np.array([[-2.0, 0.0], [0.0, 0.0], [2.0, 0.0]]))
    assert p[0] < 0.1 < p[1] < 0.9 < p[2]


def test_crc_threshold_by_hand():
    s, w = np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.0, 0.0, 1.0, 1.0])
    # bound(τ) = #wrong acted/(n+1) + 1/(n+1): 0.2, 0.2, 0.4, 0.6
    assert crc_threshold(s, w, 0.25) == 0.2
    assert crc_threshold(s, w, 0.4) == 0.3
    assert crc_threshold(s, w, 0.1) == -math.inf
    assert crc_threshold(np.array([]), np.array([]), 0.5) == -math.inf


def test_thresholds_sit_on_score_boundaries():
    s, w = np.array([0.1, 0.2, 0.2, 0.3]), np.array([0.0, 0.0, 1.0, 0.0])
    # τ=0.2 acts on both tied rows (one wrong): bound 0.4 > 0.3, so the tie is not split
    assert crc_threshold(s, w, 0.3) == 0.1


def test_binom_cdf_matches_direct_sum():
    n, p = 12, 0.3
    direct = [sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1)) for k in range(n + 1)]
    assert [binom_cdf(k, n, p) for k in range(n + 1)] == pytest.approx(direct)
    assert binom_cdf(-1, n, p) == 0.0 and binom_cdf(n, n, p) == 1.0


def test_ltt_is_at_most_as_liberal_as_crc():
    rng = np.random.default_rng(2)
    s = rng.random(600)
    w = (rng.random(600) < s).astype(float)  # error probability rises with the score
    for eps in (0.05, 0.1, 0.2):
        assert ltt_threshold(s, w, eps, 0.1) <= crc_threshold(s, w, eps)


def test_ltt_rejects_only_while_binomial_test_passes():
    s = np.arange(1, 201) / 200
    w = np.r_[np.zeros(100), np.ones(100)]
    tau = ltt_threshold(s, w, 0.05, 0.1)
    k = int(round(tau * 200))  # acting on the k lowest scores, all correct up to 100
    assert k >= 100 and binom_cdf(k - 100, 200, 0.05) <= 0.1 < binom_cdf(k - 99, 200, 0.05)


@pytest.mark.parametrize("eps", [0.05, 0.1])
def test_learned_risk_controls_error_and_beats_risk_conformal(data, eps):
    (cal_p, cal_g), (test_p, test_g) = data
    learned, risk = LearnedRisk(eps), RiskConformal(eps)
    learned.fit(cal_p, cal_g)
    risk.fit(cal_p, cal_g)
    err, auto = rates(learned, test_p, test_g)
    assert err <= eps + 0.015
    assert auto >= rates(risk, test_p, test_g)[1] - 0.02  # the synthetic error is a function of p_top: no loss


@pytest.mark.parametrize("eps", [0.05, 0.1])
def test_learned_pac_is_more_conservative(data, eps):
    (cal_p, cal_g), (test_p, test_g) = data
    crc, pac = LearnedRisk(eps), LearnedRisk(eps, delta=0.1)
    crc.fit(cal_p, cal_g)
    pac.fit(cal_p, cal_g)
    assert pac.name == f"learned_pac@{eps}" and pac.tau <= crc.tau
    assert rates(pac, test_p, test_g)[0] <= eps


def test_decisions_match_prediction_sets(data):
    (cal_p, cal_g), (test_p, _) = data
    g = LearnedRisk(0.1)
    g.fit(cal_p, cal_g)
    for p in test_p[:300]:
        d, c = g.decide(p), g.prediction_set(p)
        if d.action == "act":
            assert c == [max(p, key=p.get)] == [d.option_id] and g.score(p) <= g.tau
        else:
            assert d.candidate_set == c and len(c) != 1 and d.reason in ("set", "full_view")


def test_one_class_calibration_escalates_everything():
    g = LearnedRisk(0.1)
    g.fit([{"a": 0.9, "b": 0.1}] * 50, ["a"] * 50)
    assert g.tau == -math.inf and g.score({"a": 0.9, "b": 0.1}) == math.inf
    assert g.decide({"a": 0.99, "b": 0.01}).action == "escalate"


def test_adaptive_learned_moves_with_feedback(data):
    (cal_p, cal_g), (test_p, test_g) = data
    g = AdaptiveLearnedRisk(0.1, gamma=0.05, margin=0.0, feedback="ipw")
    g.fit(cal_p, cal_g)
    assert g.target == 0.1 and g._tau() == crc_threshold(g.oof, g.wrong, 0.1)
    acted = next(p for p in test_p if len(g.prediction_set(p)) == 1)
    wrong_gold = min(acted, key=acted.get)
    before = g.alpha_raw
    g.update(acted, wrong_gold, weight=5.0)  # an audited error
    assert g.alpha_raw == pytest.approx(before + 0.05 * (0.1 - 5.0))
    assert g.alpha == g.alpha_min and g._tau() == -math.inf  # below the floor: escalate everything
    for p in test_p[:200]:
        g.update(p, "never", weight=0.0)  # ipw: unlabelled steps only drift α up
    assert g.alpha == pytest.approx(g.alpha_max)
    g.alpha_raw = 0.0
    assert g._tau() == -math.inf and g.decide(acted).action == "escalate"


def test_adaptive_learned_doubly_robust_estimate(data):
    (cal_p, cal_g), (test_p, _) = data
    g = AdaptiveLearnedRisk(0.1)  # defaults: dr, margin 0.2, gamma 0.005
    g.fit(cal_p, cal_g)
    assert g.target == pytest.approx(0.08) and g.alpha_raw == pytest.approx(0.08)
    acted = next(p for p in test_p if len(g.prediction_set(p)) == 1)
    top, e = max(acted, key=acted.get), g.score(acted)
    assert g.loss_estimate(acted, "x", 0.0) == pytest.approx(e)  # unaudited: the model's error estimate
    assert g.loss_estimate(acted, top, 5.0) == pytest.approx(e - 5 * e)  # audited, correct
    assert g.loss_estimate(acted, "x", 5.0) == pytest.approx(e + 5 * (1 - e))  # audited, wrong
    assert g.loss_estimate(acted, "x", 1.0) == 1.0  # fully labelled: the loss itself
    escalated = next(p for p in test_p if len(g.prediction_set(p)) != 1)
    assert g.loss_estimate(escalated, "x", 1.0) == 0.0
    before = g.alpha_raw
    g.update(acted, "x", 0.0)
    assert g.alpha_raw == pytest.approx(before + 0.005 * (0.08 - e))


@pytest.mark.parametrize("kw", [{"feedback": "aipw"}, {"margin": 1.0}, {"margin": -0.1}])
def test_adaptive_learned_rejects_bad_options(kw):
    with pytest.raises(ValueError):
        AdaptiveLearnedRisk(0.1, **kw)
