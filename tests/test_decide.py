import json

import numpy as np
import pytest

from surety.decide import MockBackend, SystemOneClient, TemperatureScaler, average_runs, noul_debiased
from surety.types import DecisionState, Option

BASE = "http://kev"
OPTS = [Option("a", "Alpha"), Option("b", "Beta"), Option("c", "Gamma")]


def state(text="ctx", opts=OPTS):
    return DecisionState(text, "pick one", list(opts))


def ans(probs, tokens=10):
    return {"answers": {"q": {"type": "choice", "choice": "a", "probabilities": probs}},
            "usage": {"input_tokens": tokens}}


def bodies(httpx_mock, suffix=""):
    return [json.loads(r.content) for r in httpx_mock.get_requests() if r.url.path.endswith(suffix)]


def test_mock_backend_returns_fn_probs():
    b = MockBackend(lambda s: {"a": 0.9, "b": 0.1})
    out = b.choice(state())
    assert out.probs == {"a": 0.9, "b": 0.1}
    assert len(b.calls) == 1


def test_choice_posts_systemone_shape(httpx_mock):
    httpx_mock.add_response(url=f"{BASE}/v1/systemone", json=ans({"a": 0.5, "b": 0.3, "c": 0.2}))
    out = SystemOneClient(BASE, api_key="k").choice(state())
    req = httpx_mock.get_requests()[0]
    body = json.loads(req.content)
    assert body["questions"]["q"]["criteria"] == {"a": "Alpha", "b": "Beta", "c": "Gamma"}
    assert body["questions"]["q"]["type"] == "choice"
    assert body["state"] == "ctx" and body["model"] == "kev-latest"
    assert req.headers["Authorization"] == "Bearer k"
    assert out.input_tokens == 10


def test_choice_renormalizes_partial_probabilities(httpx_mock):
    httpx_mock.add_response(url=f"{BASE}/v1/systemone", json=ans({"a": 0.6}))
    out = SystemOneClient(BASE).choice(state(opts=OPTS[:2]))
    assert out.probs == {"a": 1.0, "b": 0.0}


def test_choice_uniform_when_no_known_ids(httpx_mock):
    httpx_mock.add_response(url=f"{BASE}/v1/systemone", json=ans({"z": 1.0}))
    out = SystemOneClient(BASE).choice(state(opts=OPTS[:2]))
    assert out.probs == {"a": 0.5, "b": 0.5}


def test_client_context_manager_closes():
    with SystemOneClient(BASE) as c:
        assert not c.http.is_closed
    assert c.http.is_closed


def test_choice_truncates_long_state(httpx_mock):
    httpx_mock.add_response(url=f"{BASE}/v1/systemone", json=ans({"a": 1.0}))
    out = SystemOneClient(BASE, max_chars=100).choice(state("x" * 500))
    assert len(bodies(httpx_mock)[0]["state"]) == 100
    assert out.truncated


def test_permute_endpoint_used_when_available(httpx_mock):
    run = {"order": ["a", "b"], "probabilities": {"a": 0.7, "b": 0.3}, "choice": "a"}
    httpx_mock.add_response(url=f"{BASE}/v1/systemone/permute", json={"runs": [run] * 3, "usage": {"input_tokens": 7}})
    out = SystemOneClient(BASE).choice(state(opts=OPTS[:2]), n_perm=3, seed=5)
    assert len(out.runs) == 3 and out.input_tokens == 7
    body = bodies(httpx_mock, "/permute")[0]
    assert body["n_perm"] == 3 and body["seed"] == 5 and body["question"] == "q"
    assert body["request"]["questions"]["q"]["criteria"] == {"a": "Alpha", "b": "Beta"}


def test_permute_falls_back_on_404(httpx_mock):
    httpx_mock.add_response(url=f"{BASE}/v1/systemone/permute", status_code=404)
    httpx_mock.add_response(url=f"{BASE}/v1/systemone", json=ans({"a": 0.5, "b": 0.3, "c": 0.2}),
                            is_reusable=True)
    c = SystemOneClient(BASE)
    out = c.choice(state(), n_perm=3, seed=1)
    orders = [list(b["questions"]["q"]["criteria"]) for b in bodies(httpx_mock, "/v1/systemone")]
    assert len(orders) == 3 and len(out.runs) == 3 and out.input_tokens == 30
    assert orders[0] == ["a", "b", "c"]
    assert len({tuple(o) for o in orders}) > 1
    c.choice(state(), n_perm=2, seed=1)
    assert len(bodies(httpx_mock, "/permute")) == 1


def test_noul_returns_probability(httpx_mock):
    httpx_mock.add_response(url=f"{BASE}/v1/systemone",
                            json={"answers": {"q": {"type": "noul", "noul": 0.8}}})
    assert SystemOneClient(BASE).noul("ctx", "is it?") == 0.8


def test_noul_debiased_averages_negation():
    vals = iter([0.8, 0.4])
    b = MockBackend()
    b.noul = lambda s, i: next(vals)
    assert noul_debiased(b, "t", "i", "not i") == pytest.approx(0.7)


def test_average_runs_renormalizes():
    out = average_runs([{"a": 3.0, "b": 1.0}, {"a": 1.0, "b": 1.0}])
    assert out == pytest.approx({"a": 4 / 6, "b": 2 / 6})


def test_temperature_fit_recovers_sharpening():
    rng = np.random.default_rng(0)
    probs, gold = [], []
    for _ in range(400):
        logits = rng.normal(size=4)
        true = np.exp(logits) / np.exp(logits).sum()
        g = int(rng.choice(4, p=true))
        sharp = np.exp(3 * logits) / np.exp(3 * logits).sum()
        probs.append({str(i): float(p) for i, p in enumerate(sharp)})
        gold.append(str(g))
    ts = TemperatureScaler()
    t = ts.fit(probs, gold)
    assert 2.0 < t < 4.5 and ts.T == t
    out = ts.apply(probs[0])
    assert sum(out.values()) == pytest.approx(1.0)
    assert max(out.values()) < max(probs[0].values())
