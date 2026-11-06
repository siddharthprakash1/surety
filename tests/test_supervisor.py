import json

from surety.supervise import MockSupervisor, Supervisor
from surety.types import DecisionState, Option


def _state():
    return DecisionState(
        "page text", "pick one", [Option("a1", "Alpha"), Option("b2", "Bravo"), Option("c3", "Charlie")]
    )


def _reply(httpx_mock, content, usage=None):
    body = {"choices": [{"message": {"content": content}}]}
    if usage is not None:
        body["usage"] = usage
    httpx_mock.add_response(url="http://sup/v1/chat/completions", json=body)


def _sup():
    return Supervisor("http://sup", "m")


def test_choose_restricts_prompt_to_candidates(httpx_mock):
    _reply(httpx_mock, "c3")
    _sup().choose(_state(), ["c3", "a1"])
    req = json.loads(httpx_mock.get_request().content)
    user = req["messages"][1]["content"]
    assert "Charlie" in user and "Alpha" in user and "Bravo" not in user
    assert user.index("c3: Charlie") < user.index("a1: Alpha")
    assert req["model"] == "m" and req["max_tokens"] == 64 and req["temperature"] == 0.0


def test_choose_parses_option_id(httpx_mock):
    _reply(httpx_mock, "Answer: b2.")
    assert _sup().choose(_state())[0] == "b2"


def test_choose_rejects_unknown_option(httpx_mock):
    _reply(httpx_mock, "z9")
    ans, usage = _sup().choose(_state())
    assert ans is None and usage.latency_ms >= 0


def test_choose_none_and_disallowed_candidate(httpx_mock):
    _reply(httpx_mock, "NONE")
    assert _sup().choose(_state())[0] is None
    _reply(httpx_mock, "b2")
    assert _sup().choose(_state(), ["a1"])[0] is None


def test_usage_from_response(httpx_mock):
    _reply(httpx_mock, "a1", {"prompt_tokens": 12, "completion_tokens": 3})
    _, u = _sup().choose(_state())
    assert (u.prompt_tokens, u.completion_tokens) == (12, 3)


def test_usage_missing_defaults_zero(httpx_mock):
    _reply(httpx_mock, "a1")
    _, u = _sup().choose(_state())
    assert (u.prompt_tokens, u.completion_tokens) == (0, 0)


def test_mock_supervisor_counts_calls():
    m = MockSupervisor("a1")
    assert m.choose(_state())[0] == "a1" and m.choose(_state())[0] == "a1"
    assert m.calls == 2


def test_reasoning_level_prefixes_system_message(httpx_mock):
    _reply(httpx_mock, "a1")
    Supervisor("http://sup", "m", reasoning="low").choose(_state())
    assert json.loads(httpx_mock.get_request().content)["messages"][0]["content"].startswith("Reasoning: low\n")


def test_empty_content_falls_back_to_reasoning_tail(httpx_mock):
    body = {"choices": [{"message": {"content": "", "reasoning": "a1 maybe " + "x" * 300 + " so b2"}}]}
    httpx_mock.add_response(url="http://sup/v1/chat/completions", json=body)
    chosen, usage = _sup().choose(_state())
    assert chosen == "b2" and usage.fallback_parsed
