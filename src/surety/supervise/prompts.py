from surety.types import DecisionState

SYSTEM = "You supervise a web agent. Reply with exactly one option id from the list, or NONE if no option fits."
SLOT_SYSTEM = "You supervise a web agent. Reply with only the text to type, nothing else."


def build_choose_prompt(state: DecisionState, candidate_ids: list[str] | None = None) -> list[dict]:
    labels = {o.id: o.label for o in state.options}
    ids = candidate_ids if candidate_ids is not None else [o.id for o in state.options]
    lines = "\n".join(f"{i}: {labels[i]}" for i in ids if i in labels)
    user = f"{state.instructions}\n\n{state.state_text}\n\n{lines}\n\nAnswer with the id only."
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def build_slot_prompt(state: DecisionState, element_id: str) -> list[dict]:
    label = next((o.label for o in state.options if o.id == element_id), element_id)
    task = state.meta.get("task", state.instructions)
    user = f"{state.state_text}\n\nGive only the text to type into {label} for task {task}"
    return [{"role": "system", "content": SLOT_SYSTEM}, {"role": "user", "content": user}]
