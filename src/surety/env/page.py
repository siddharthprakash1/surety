from dataclasses import dataclass
from typing import Literal


@dataclass
class Element:
    id: str
    role: str
    name: str
    value: str = ""


@dataclass
class Page:
    url: str
    title: str
    text: str
    elements: list[Element]


@dataclass
class Action:
    kind: Literal["click", "type", "select", "scroll", "back", "done"]
    element_id: str | None = None
    text: str | None = None
