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
