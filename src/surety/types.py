from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class Option:
    id: str
    label: str


@dataclass
class DecisionState:
    state_text: str
    instructions: str
    options: list[Option]
    meta: dict = field(default_factory=dict)


@dataclass
class ChoiceAnswer:
    probs: dict[str, float]
    runs: list[dict[str, float]]
    latency_ms: float
    input_tokens: int
    truncated: bool = False

    def top(self) -> tuple[str, float]:
        return max(self.probs.items(), key=lambda kv: kv[1])


@dataclass
class GateDecision:
    action: Literal["act", "escalate"]
    option_id: str | None
    candidate_set: list[str]
    reason: str


@dataclass
class Instance:
    id: str
    site: str
    task_type: str
    question_type: str
    options: list[str]
    gold: int
    context: str
    question: str
    num_tokens: int
    context_format: str
