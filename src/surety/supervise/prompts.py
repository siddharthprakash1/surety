from surety.types import DecisionState

SYSTEM = "You supervise a web agent. Reply with exactly one option id from the list, or NONE if no option fits."


def build_choose_prompt(state: DecisionState, candidate_ids: list[str] | None = None) -> list[dict]:
    labels = {o.id: o.label for o in state.options}
    ids = candidate_ids if candidate_ids is not None else [o.id for o in state.options]
    lines = "\n".join(f"{i}: {labels[i]}" for i in ids if i in labels)
    user = f"{state.instructions}\n\n{state.state_text}\n\n{lines}\n\nAnswer with the id only."
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
