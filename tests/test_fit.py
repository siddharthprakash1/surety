import math
import random

import pytest

from surety.decide import TemperatureScaler
from surety.gate import RiskConformal
from surety.loop.fit import fit_gate


def _rows(n=200, seed=0):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        gold = rng.choice("abc")
        top = gold if rng.random() < 0.7 else rng.choice("abc")
        w = {o: (4.0 if o == top else rng.random()) for o in "abc"}
        z = sum(w.values())
        rows.append({"split": "cal" if i < n // 2 else "test_in", "gold_id": gold,
                     "probs": {o: v / z for o, v in w.items()}})
    return rows


@pytest.mark.parametrize("spec,probs,action", [
    ("supervisor", {"a": 0.99, "b": 0.01}, "escalate"),
    ("executor", {"a": 0.4, "b": 0.6}, "act"),
    ("escalate_on_none", {"a": 0.9, "none_of_these": 0.1}, "act"),
    ("escalate_on_none", {"a": 0.1, "none_of_these": 0.9}, "escalate"),
])
def test_baselines_need_no_collection(spec, probs, action):
    gate, scaler = fit_gate(spec)
    assert scaler is None and gate.decide(probs).action == action


def test_fitted_gate_uses_temperature_scaled_cal_rows():
    rows = _rows()
    gate, scaler = fit_gate("risk:0.1", rows)
    cal = [r for r in rows if r["split"] == "cal"]
    ref_t = TemperatureScaler()
    ref_t.fit([r["probs"] for r in cal], [r["gold_id"] for r in cal])
    ref = RiskConformal(0.1)
    ref.fit([ref_t.apply(r["probs"]) for r in cal], [r["gold_id"] for r in cal])
    assert scaler.T == ref_t.T and gate.q == ref.q < math.inf


def test_adaptive_spec_sets_gamma_and_fitted_spec_needs_rows():
    assert fit_gate("arisk:0.05", _rows(), gamma=0.05)[0].gamma == 0.05
    with pytest.raises(ValueError):
        fit_gate("risk:0.1")


@pytest.mark.parametrize("spec", ["per_choice:0.05", "risk:5", "fitted:0", "fixed:1.5"])
def test_rejected_specs(spec):
    with pytest.raises(ValueError):
        fit_gate(spec, _rows())


def test_learned_specs():
    from surety.gate import AdaptiveLearnedRisk, LearnedRisk

    crc, pac, ada = (fit_gate(s, _rows(), gamma=0.05)[0] for s in ("learned:0.1", "learned_pac:0.1", "alearned:0.1"))
    assert type(crc) is LearnedRisk and crc.delta is None and crc.models
    assert type(pac) is LearnedRisk and pac.delta == 0.1 and pac.tau <= crc.tau
    assert isinstance(ada, AdaptiveLearnedRisk) and ada.gamma == 0.05 and ada.models
