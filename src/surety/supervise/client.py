import json
import re
import time
from dataclasses import dataclass

import httpx

from surety.supervise.prompts import build_choose_prompt, build_slot_prompt
from surety.types import DecisionState


@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    fallback_parsed: bool = False
    abstained: bool = False


def _pattern(allowed: list[str]) -> str:
    return "|".join(re.escape(i) for i in sorted(allowed, key=len, reverse=True))


def _parse(text: str, allowed: list[str], last: bool = False) -> str | None:
    if not allowed:
        return None
    found = re.findall(rf"(?<![\w-])(?:{_pattern(allowed)})(?![\w-])", text)
    return (found[-1] if last else found[0]) if found else None


def _is_none(text: str) -> bool:
    return re.search(r"(?<![\w-])NONE(?![\w-])", text) is not None


def _id_schema(allowed: list[str]) -> dict:
    return {"name": "answer", "strict": True,
            "schema": {"type": "object", "properties": {"id": {"type": "string", "enum": [*allowed, "NONE"]}},
                       "required": ["id"], "additionalProperties": False}}


def _json_id(text: str) -> str | None:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value.get("id") if isinstance(value, dict) and isinstance(value.get("id"), str) else None


class Supervisor:
    """json_mode: the reply is constrained to {"id": <allowed id> | "NONE"} (OpenAI-style response_format
    json_schema; OpenAI, vLLM, Ollama, OpenRouter accept it); free-text parsing remains the fallback."""

    def __init__(self, base_url: str, model: str, timeout: float = 300.0,
                 max_tokens: int = 64, temperature: float | None = 0.0, reasoning: str | None = None,
                 api_key: str | None = None, json_mode: bool = False, token_param: str = "max_tokens"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.reasoning = reasoning
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.json_mode = json_mode
        self.token_param = token_param  # max_completion_tokens for OpenAI reasoning models (which also reject temperature)

    def _complete(self, messages: list[dict], schema: dict | None = None) -> tuple[str, Usage]:
        if self.reasoning:
            messages = [{**messages[0], "content": f"Reasoning: {self.reasoning}\n{messages[0]['content']}"}, *messages[1:]]
        payload = {"model": self.model, "messages": messages, self.token_param: self.max_tokens}
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if schema:
            payload["response_format"] = {"type": "json_schema", "json_schema": schema}
        t0 = time.perf_counter()
        r = httpx.post(f"{self.base_url}/v1/chat/completions", json=payload, headers=self.headers,
                       timeout=self.timeout)
        latency = (time.perf_counter() - t0) * 1000
        r.raise_for_status()
        body = r.json()
        u = body.get("usage") or {}
        usage = Usage(u.get("prompt_tokens", 0), u.get("completion_tokens", 0), latency)
        choice = body["choices"][0]
        msg = choice["message"]
        text = msg.get("content") or ""
        if not text.strip():
            think = msg.get("reasoning") or msg.get("reasoning_content") or ""
            if think.strip() and choice.get("finish_reason") != "length":
                usage.fallback_parsed = True
                text = think[-200:]
        return text, usage

    def choose(self, state: DecisionState, candidate_ids: list[str] | None = None) -> tuple[str | None, Usage]:
        allowed = candidate_ids if candidate_ids is not None else [o.id for o in state.options]
        text, usage = self._complete(build_choose_prompt(state, candidate_ids),
                                     _id_schema(allowed) if self.json_mode else None)
        if self.json_mode and not usage.fallback_parsed and (got := _json_id(text)) is not None:
            usage.abstained = got == "NONE"
            return (got if got in allowed else None), usage
        chosen = _parse(text, allowed, last=usage.fallback_parsed)
        # a NONE in recovered reasoning text is not an answer
        usage.abstained = chosen is None and not usage.fallback_parsed and _is_none(text)
        return chosen, usage

    def slot_value(self, state: DecisionState, element_id: str) -> tuple[str | None, Usage]:
        text, usage = self._complete(build_slot_prompt(state, element_id))
        if usage.fallback_parsed:
            return None, usage
        return (text.strip().splitlines() or [""])[0].strip() or None, usage


class MockSupervisor:
    model, json_mode = "mock", False

    def __init__(self, answer: str | None, slots: dict[str, str] | None = None):
        self.answer = answer
        self.slots = slots or {}
        self.calls = 0
        self.candidates: list[list[str] | None] = []

    def choose(self, state: DecisionState, candidate_ids: list[str] | None = None) -> tuple[str | None, Usage]:
        self.calls += 1
        self.candidates.append(candidate_ids)
        return self.answer, Usage(0, 0, 0.0)

    def slot_value(self, state: DecisionState, element_id: str) -> tuple[str | None, Usage]:
        self.calls += 1
        return self.slots.get(element_id), Usage(0, 0, 0.0)
