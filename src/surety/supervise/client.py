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


def _parse(text: str, allowed: list[str]) -> str | None:
    if not allowed:
        return None
    ids = sorted(allowed, key=len, reverse=True)
    pat = "|".join(re.escape(i) for i in ids)
    m = re.search(rf"(?<![\w-])(?:{pat})(?![\w-])", text)
    return m.group(0) if m else None


class Supervisor:
    def __init__(self, base_url: str, model: str, timeout: float = 300.0,
                 max_tokens: int = 64, temperature: float = 0.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _complete(self, messages: list[dict]) -> tuple[str, Usage]:
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
        return body["choices"][0]["message"]["content"] or "", usage

    def choose(self, state: DecisionState, candidate_ids: list[str] | None = None) -> tuple[str | None, Usage]:
        allowed = candidate_ids if candidate_ids is not None else [o.id for o in state.options]
        text, usage = self._complete(build_choose_prompt(state, candidate_ids))
        return _parse(text, allowed), usage

    def slot_value(self, state: DecisionState, element_id: str) -> tuple[str | None, Usage]:
        text, usage = self._complete(build_slot_prompt(state, element_id))
        return text.strip() or None, usage


class MockSupervisor:
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
