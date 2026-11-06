from typing import Callable

from surety.types import ChoiceAnswer, DecisionState


class MockBackend:
    name = "mock"

    def __init__(self, fn: Callable[[DecisionState], dict[str, float]] | None = None,
                 noul_value: float = 0.5):
        self.fn = fn
        self.noul_value = noul_value
        self.calls: list[tuple] = []

    def choice(self, state: DecisionState, n_perm: int = 1, seed: int = 0) -> ChoiceAnswer:
        self.calls.append(("choice", state, n_perm, seed))
        if self.fn:
            probs = self.fn(state)
        else:
            probs = {o.id: 1 / len(state.options) for o in state.options}
        return ChoiceAnswer(probs=probs, runs=[probs] * n_perm, latency_ms=0.0, input_tokens=0)

    def noul(self, state_text: str, instructions: str) -> float:
        self.calls.append(("noul", state_text, instructions))
        return self.noul_value
