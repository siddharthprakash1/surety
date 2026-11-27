import json
import time

import httpx
import pytest

from surety.supervise import MockSupervisor, RateLimited, Supervisor
from surety.supervise.client import _seconds, _wait_hint
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


def test_fallback_takes_last_id_and_skips_truncated(httpx_mock):
    reply = {"choices": [{"message": {"content": "", "reasoning": "a1 no, b2 hmm, finally c3"}}]}
    httpx_mock.add_response(url="http://sup/v1/chat/completions", json=reply)
    assert _sup().choose(_state())[0] == "c3"
    reply["choices"][0]["finish_reason"] = "length"
    httpx_mock.add_response(url="http://sup/v1/chat/completions", json=reply)
    chosen, usage = _sup().choose(_state())
    assert chosen is None and not usage.fallback_parsed and not usage.abstained


def test_none_in_recovered_reasoning_is_not_abstention(httpx_mock):
    body = {"choices": [{"message": {"content": "", "reasoning": "reply with exactly its id, not NONE."}}]}
    httpx_mock.add_response(url="http://sup/v1/chat/completions", json=body)
    chosen, usage = _sup().choose(_state())
    assert chosen is None and usage.fallback_parsed and not usage.abstained


def test_explicit_none_is_abstention(httpx_mock):
    _reply(httpx_mock, "NONE")
    chosen, usage = _sup().choose(_state())
    assert chosen is None and usage.abstained


def test_slot_value_prompt_and_strip(httpx_mock):
    _reply(httpx_mock, "  running shoes\n")
    s = _state()
    s.meta["task"] = "find shoes"
    text, usage = _sup().slot_value(s, "b2")
    assert text == "running shoes" and usage.latency_ms >= 0
    user = json.loads(httpx_mock.get_request().content)["messages"][1]["content"]
    assert "Give only the text to type into Bravo for task find shoes" in user


def test_slot_value_empty_is_none(httpx_mock):
    _reply(httpx_mock, "")
    assert _sup().slot_value(_state(), "a1")[0] is None


def test_json_mode_sends_schema_and_key(httpx_mock):
    _reply(httpx_mock, '{"id": "c3"}')
    chosen, usage = Supervisor("http://sup", "m", api_key="k", json_mode=True).choose(_state(), ["c3", "a1"])
    req = httpx_mock.get_request()
    fmt = json.loads(req.content)["response_format"]
    assert req.headers["Authorization"] == "Bearer k" and fmt["type"] == "json_schema"
    assert fmt["json_schema"]["schema"]["properties"]["id"]["enum"] == ["c3", "a1", "NONE"]
    assert chosen == "c3" and not usage.abstained


def test_json_mode_none_abstains_and_disallowed_id_is_unparsed(httpx_mock):
    sup = Supervisor("http://sup", "m", json_mode=True)
    _reply(httpx_mock, '{"id": "NONE"}')
    assert (lambda c, u: (c, u.abstained))(*sup.choose(_state())) == (None, True)
    _reply(httpx_mock, '{"id": "b2"}')
    assert (lambda c, u: (c, u.abstained))(*sup.choose(_state(), ["a1"])) == (None, False)


def test_json_mode_falls_back_to_text_parse(httpx_mock):
    _reply(httpx_mock, "I pick b2")
    assert Supervisor("http://sup", "m", json_mode=True).choose(_state())[0] == "b2"


def test_reasoning_model_payload_omits_temperature(httpx_mock):
    _reply(httpx_mock, "a1")
    Supervisor("http://sup", "m", max_tokens=99, temperature=None, token_param="max_completion_tokens").choose(_state())
    req = json.loads(httpx_mock.get_request().content)
    assert req["max_completion_tokens"] == 99 and "max_tokens" not in req and "temperature" not in req


def test_default_sends_no_response_format_or_key(httpx_mock):
    _reply(httpx_mock, "a1")
    _sup().choose(_state())
    req = httpx_mock.get_request()
    assert "response_format" not in json.loads(req.content) and "Authorization" not in req.headers


def test_slot_value_never_from_reasoning(httpx_mock):
    body = {"choices": [{"message": {"content": "", "reasoning": "the user wants shoes so type shoes"}}]}
    httpx_mock.add_response(url="http://sup/v1/chat/completions", json=body)
    text, usage = _sup().slot_value(_state(), "a1")
    assert text is None and usage.fallback_parsed


def test_slot_value_first_line_only(httpx_mock):
    _reply(httpx_mock, "\n shoes \nbecause the task says so")
    assert _sup().slot_value(_state(), "a1")[0] == "shoes"


def test_mock_supervisor_slots_and_candidates():
    m = MockSupervisor("a1", slots={"b2": "x"})
    assert m.slot_value(_state(), "b2")[0] == "x" and m.slot_value(_state(), "a1")[0] is None
    m.choose(_state(), ["a1"])
    assert m.candidates == [["a1"]] and m.calls == 3


URL = "http://sup/v1/chat/completions"


def _sup_waits(**kw):
    sup, waits = Supervisor("http://sup", "m", **kw), []
    sup.sleep = waits.append
    return sup, waits


def test_rate_limit_waits_as_asked_then_answers(httpx_mock):
    httpx_mock.add_response(url=URL, status_code=429, headers={"retry-after": "7"})
    httpx_mock.add_response(url=URL, status_code=429, json={"error": {"message": "Please try again in 1m2.5s."}})
    _reply(httpx_mock, "b2")
    sup, waits = _sup_waits()
    assert sup.choose(_state())[0] == "b2" and waits == [7.0, 62.5]


def test_daily_cap_stops_without_waiting(httpx_mock):
    httpx_mock.add_response(url=URL, status_code=429, headers={"retry-after": "3600"})
    sup, waits = _sup_waits(max_wait=900)
    with pytest.raises(RateLimited) as e:
        sup.choose(_state())
    assert e.value.wait == 3600 and waits == []


def test_waits_add_up_to_max_wait(httpx_mock):
    for _ in range(3):
        httpx_mock.add_response(url=URL, status_code=429, headers={"retry-after": "400"})
    sup, waits = _sup_waits(max_wait=900)
    with pytest.raises(RateLimited):
        sup.choose(_state())
    assert waits == [400.0, 400.0]


def test_server_errors_retry_then_raise_with_the_endpoint_message(httpx_mock):
    for _ in range(4):
        httpx_mock.add_response(url=URL, status_code=503, json={"error": {"message": "busy for org_ab12"}})
    sup, waits = _sup_waits()
    with pytest.raises(httpx.HTTPStatusError, match=r"busy for org_\*$"):
        sup.choose(_state())
    assert waits == [2.0, 4.0, 8.0]


def test_too_large_and_bad_requests_fail_at_once(httpx_mock):
    too_large = {"error": {"message": "Request too large for model on tokens per minute (TPM): Limit 8000"}}
    httpx_mock.add_response(url=URL, status_code=413, json=too_large)
    httpx_mock.add_response(url=URL, status_code=429, json=too_large)
    httpx_mock.add_response(url=URL, status_code=400, json={"error": {"message": "reasoning_effort unsupported"}})
    sup, waits = _sup_waits()
    for pattern in ("413.*Limit 8000", "429.*Limit 8000", "reasoning_effort unsupported"):
        with pytest.raises(httpx.HTTPStatusError, match=pattern):
            sup.choose(_state())
    assert waits == []


def test_dropped_connection_is_retried(httpx_mock):
    httpx_mock.add_exception(httpx.ConnectError("refused"), url=URL)
    _reply(httpx_mock, "a1", {"prompt_tokens": 5, "completion_tokens": 1})
    sup, waits = _sup_waits()
    label, usage = sup.choose(_state())
    assert (label, usage.prompt_tokens, waits) == ("a1", 5, [2.0])


def test_params_join_the_request(httpx_mock):
    _reply(httpx_mock, "a1")
    Supervisor("http://sup", "m", params={"reasoning_effort": "low"}).choose(_state())
    assert json.loads(httpx_mock.get_request().content)["reasoning_effort"] == "low"


def test_wait_hints():
    assert _seconds("1m2.5s") == 62.5 and _seconds("450ms") == pytest.approx(0.45) and _seconds("2h") == 7200
    assert _seconds("soon") is None
    assert _wait_hint(httpx.Response(429, headers={"x-ratelimit-reset": "30"})) == 30.0
    in_2_min = str(int((time.time() + 120) * 1000))
    assert 100 < _wait_hint(httpx.Response(429, headers={"x-ratelimit-reset": in_2_min})) <= 120
    assert _wait_hint(httpx.Response(429, text="slow down")) is None
