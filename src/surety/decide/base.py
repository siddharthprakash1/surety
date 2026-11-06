from typing import Protocol

from surety.types import ChoiceAnswer, DecisionState


class Backend(Protocol):
    name: str

    def choice(self, state: DecisionState, n_perm: int = 1, seed: int = 0) -> ChoiceAnswer: ...

    def noul(self, state_text: str, instructions: str) -> float: ...
