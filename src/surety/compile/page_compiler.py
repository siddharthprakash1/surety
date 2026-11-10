from surety.env.page import Element, Page
from surety.types import DecisionState, Option

NONE_OF_THESE = "none_of_these"
_NONE = Option(NONE_OF_THESE, "None of these; the page needs a different action (scroll, back, or the task is done)")


def _label(e: Element) -> str:
    return f"{e.role}: {e.name}" + (f" [value={e.value}]" if e.value else "")


def compile_page(page: Page, task: str, history: list[str], max_elements: int = 40,
                 max_text: int = 1500) -> DecisionState:
    options = [Option(e.id, _label(e)) for e in page.elements[:max_elements]]
    hist = "\n".join(history[-8:])
    state_text = (f"TASK\n{task}\n\nURL\n{page.url}\n\nTITLE\n{page.title}\n\n"
                  f"PAGE TEXT\n{page.text[:max_text]}\n\nHISTORY\n{hist}")
    return DecisionState(
        state_text=state_text,
        instructions=f"Which element should the agent act on next to accomplish: {task}",
        options=[*options, _NONE],
        meta={"url": page.url, "n_elements": len(page.elements), "task": task},
    )
