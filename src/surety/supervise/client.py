import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import httpx

from surety.supervise.prompts import build_choose_prompt, build_slot_prompt
from surety.types import DecisionState

SERVER_ERRORS = frozenset({500, 502, 503, 504})
TRIES = 3  # retries after a server error or a dropped connection


class RateLimited(RuntimeError):
    """The endpoint asked for a longer pause than max_wait allows (a daily cap): re-run the same command later."""

    def __init__(self, wait: float | None):
        self.wait = wait
        super().__init__("rate limited" + (f"; the endpoint asks for {wait:.0f} s" if wait is not None else ""))


def _seconds(text: str) -> float | None:
    """'1m2.5s', '450ms', '2h3m' -> seconds."""
    parts = re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s)", text)
    return sum(float(v) * {"ms": 1e-3, "h": 3600.0, "m": 60.0, "s": 1.0}[u] for v, u in parts) if parts else None


def _wait_hint(r: httpx.Response) -> float | None:
    """The pause a 429 asks for: retry-after (seconds or a date), x-ratelimit-reset (epoch; OpenRouter), else "try
    again in 1m2.5s" in the body (Groq, OpenAI)."""
    now = datetime.now(timezone.utc)
    if h := r.headers.get("retry-after", "").strip():
        try:
            return max(0.0, float(h))
        except ValueError:
            try:
                return max(0.0, (parsedate_to_datetime(h) - now).total_seconds())
            except (TypeError, ValueError):
                pass
    if (h := r.headers.get("x-ratelimit-reset", "").strip()).isdigit():
        v = float(h)  # epoch milliseconds, epoch seconds, or seconds from now
        if v > 1e12:
            v /= 1000
        return max(0.0, v - now.timestamp()) if v > 1e9 else v
    m = re.search(r"try again in ((?:\d+(?:\.\d+)?(?:ms|h|m|s))+)", r.text)
    return _seconds(m.group(1)) if m else None


def _raise(r: httpx.Response) -> None:
    """raise_for_status with the endpoint's own message (a 400 names the field it rejects), org ids dropped."""
    if r.is_success:
        return
    try:
        body = r.json()
    except ValueError:
        body = r.text
    err = body.get("error", body) if isinstance(body, dict) else body
    detail = re.sub(r"\borg_\w+", "org_*", str(err.get("message", "") if isinstance(err, dict) else err))[:300]
    try:
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise httpx.HTTPStatusError(f"{e}: {detail}" if detail else str(e), request=e.request, response=r) from None


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
    json_schema; OpenAI, vLLM, Ollama, OpenRouter and Groq's gpt-oss accept it); free-text parsing remains the fallback.
    params: extra request fields (Groq's gpt-oss takes its reasoning level as reasoning_effort, not from the prompt).
    max_wait: seconds one call may spend waiting out rate limits before it raises RateLimited."""

    def __init__(self, base_url: str, model: str, timeout: float = 300.0,
                 max_tokens: int = 64, temperature: float | None = 0.0, reasoning: str | None = None,
                 api_key: str | None = None, json_mode: bool = False, token_param: str = "max_tokens",
                 params: dict | None = None, max_wait: float = 900.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.reasoning = reasoning
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.json_mode = json_mode
        self.token_param = token_param  # max_completion_tokens for OpenAI reasoning models (which also reject temperature)
        self.params = dict(params or {})
        self.max_wait = max_wait
        self.sleep = time.sleep

    def _post(self, payload: dict) -> tuple[httpx.Response, float]:
        """A 429 waits as long as the endpoint asks while the waits add up to at most max_wait, else RateLimited (a
        daily cap); a request too large for the per-minute cap fails at once; a server error or a dropped connection is
        retried TRIES times with backoff. Returns the answered attempt and its latency in ms."""
        waited, limits, fails = 0.0, 0, 0
        while True:
            t0 = time.perf_counter()
            try:
                r = httpx.post(f"{self.base_url}/v1/chat/completions", json=payload, headers=self.headers,
                               timeout=self.timeout)
            except httpx.TransportError:
                if fails == TRIES:
                    raise
                r = None
            latency = (time.perf_counter() - t0) * 1000
            if r is not None and r.status_code == 429 and "request too large" not in r.text.lower():
                limits += 1
                hint = _wait_hint(r)
                wait = max(1.0, hint if hint is not None else min(60.0, 2.0 ** limits))  # >= 1 s: no busy loop
                if waited + wait > self.max_wait:
                    raise RateLimited(hint)
            elif r is None or r.status_code in SERVER_ERRORS:
                if fails == TRIES:
                    _raise(r)
                fails += 1
                wait = 2.0 ** fails
            else:
                _raise(r)
                return r, latency
            self.sleep(wait)
            waited += wait

    def _complete(self, messages: list[dict], schema: dict | None = None) -> tuple[str, Usage]:
        if self.reasoning:
            messages = [{**messages[0], "content": f"Reasoning: {self.reasoning}\n{messages[0]['content']}"}, *messages[1:]]
        payload = {"model": self.model, "messages": messages, self.token_param: self.max_tokens}
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if schema:
            payload["response_format"] = {"type": "json_schema", "json_schema": schema}
        payload.update(self.params)
        r, latency = self._post(payload)
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
