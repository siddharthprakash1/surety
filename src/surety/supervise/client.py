import re
import time
from dataclasses import dataclass

import httpx

from surety.supervise.prompts import build_choose_prompt
from surety.types import DecisionState


@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    fallback_parsed: bool = False


def _parse(text: str, allowed: list[str]) -> str | None:
    if not allowed:
        return None
    ids = sorted(allowed, key=len, reverse=True)
    pat = "|".join(re.escape(i) for i in ids)
    m = re.search(rf"(?<![\w-])(?:{pat})(?![\w-])", text)
    return m.group(0) if m else None


class Supervisor:
    def __init__(self, base_url: str, model: str, timeout: float = 300.0,
                 max_tokens: int = 64, temperature: float = 0.0, reasoning: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.reasoning = reasoning

    def choose(self, state: DecisionState, candidate_ids: list[str] | None = None) -> tuple[str | None, Usage]:
        allowed = candidate_ids if candidate_ids is not None else [o.id for o in state.options]
        messages = build_choose_prompt(state, candidate_ids)
        if self.reasoning:
            messages[0]["content"] = f"Reasoning: {self.reasoning}\n{messages[0]['content']}"
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        t0 = time.perf_counter()
        r = httpx.post(f"{self.base_url}/v1/chat/completions", json=payload, timeout=self.timeout)
        latency = (time.perf_counter() - t0) * 1000
        r.raise_for_status()
        body = r.json()
        u = body.get("usage") or {}
        usage = Usage(u.get("prompt_tokens", 0), u.get("completion_tokens", 0), latency)
        msg = body["choices"][0]["message"]
        text = msg.get("content") or ""
        if not text.strip():
            think = msg.get("reasoning") or msg.get("reasoning_content") or ""
            usage.fallback_parsed = bool(think.strip())
            text = think[-200:]
        return _parse(text, allowed), usage


class MockSupervisor:
    def __init__(self, answer: str | None):
        self.answer = answer
        self.calls = 0

    def choose(self, state: DecisionState, candidate_ids: list[str] | None = None) -> tuple[str | None, Usage]:
        self.calls += 1
        return self.answer, Usage(0, 0, 0.0)
