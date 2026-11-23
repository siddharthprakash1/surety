from typing import Protocol

from surety.env.page import Action, Page


class EnvClosedError(RuntimeError):
    """The browser or page died mid-episode: the episode could not run (retried on resume), unlike an action the
    page rejected, which is an ordinary failed episode."""


class Env(Protocol):
    def observe(self) -> Page: ...

    def act(self, action: Action) -> None: ...

    def close(self) -> None: ...
