from typing import Protocol

from surety.env.page import Action, Page


class Env(Protocol):
    def observe(self) -> Page: ...

    def act(self, action: Action) -> None: ...

    def close(self) -> None: ...
