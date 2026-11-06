from dataclasses import asdict, dataclass
from functools import partial

from surety.compile import NONE_OF_THESE, compile_page
from surety.decide import Backend
from surety.env import Action, Env
from surety.gate import GatePolicy
from surety.trace import Trace
from surety.types import DecisionState

TYPE_ROLES = {"input", "textbox", "textarea", "searchbox"}
SELECT_ROLES = {"select", "combobox"}


@dataclass
class Episode:
    task: str
    steps: list[dict]
    stop_reason: str
    supervisor_calls: int
    executor_calls: int


def _action(option_id: str | None, roles: dict[str, str], state: DecisionState, supervisor) -> Action | None:
    if option_id in (NONE_OF_THESE, None):
        return None
    role = roles.get(option_id)
    if role in TYPE_ROLES:
        return Action("type", option_id, supervisor.slot_value(state, option_id))
    if role in SELECT_ROLES:
        return Action("select", option_id, supervisor.slot_value(state, option_id))
    return Action("click", option_id)


def _ask(supervisor, state: DecisionState, ep: Episode, row: dict, candidates: list[str] | None) -> str | None:
    choice, usage = supervisor.choose(state, candidates)
    ep.supervisor_calls += 1
    row["supervisor"].append(asdict(usage))
    return choice


def _key(a: Action) -> tuple:
    return a.kind, a.element_id, a.text


def run(task: str, env: Env, backend: Backend, policy: GatePolicy, supervisor, trace: Trace | None = None,
        max_steps: int = 30, n_perm: int = 1) -> Episode:
    ep = Episode(task, [], "max_steps", 0, 0)
    history: list[str] = []
    last, streak, nones = None, 0, 0
    for i in range(max_steps):
        page = env.observe()
        state = compile_page(page, task, history)
        labels = {o.id: o.label for o in state.options}
        roles = {e.id: e.role for e in page.elements}
        if hasattr(policy, "set_labels"):
            policy.set_labels(labels)
        answer = backend.choice(state, n_perm)
        ep.executor_calls += 1
        d = policy.decide(answer.probs)
        row = {"step": i, "url": page.url, "n_options": len(state.options), "probs": answer.probs,
               "decision": d.action, "reason": d.reason, "candidate_set": d.candidate_set,
               "executor_ms": answer.latency_ms, "supervisor": []}
        ask = partial(_ask, supervisor, state, ep, row)

        stop, chosen, action = None, None, None
        if d.reason.startswith("stop:"):
            stop = d.reason
        else:
            chosen = d.option_id
            if d.action == "escalate":
                chosen = ask(d.candidate_set or None)
                if chosen in (NONE_OF_THESE, None):
                    chosen = ask(None)
            action = _action(chosen, roles, state, supervisor)
            same = streak if action and _key(action) == last else 0
            if same >= 3:
                stop, action = "stuck", None
            elif same == 2:
                chosen = ask(None)
                action = _action(chosen, roles, state, supervisor)
            if not stop and action is None:
                nones = nones + 1 if chosen == NONE_OF_THESE else 0
                stop = "done" if nones == 2 else None
                action = Action("done" if stop else "scroll")
            elif action:
                nones = 0

        if action:
            env.act(action)
            streak = streak + 1 if _key(action) == last else 1
            last = _key(action)
            history.append(f"{action.kind} {labels.get(action.element_id, '')}".strip()
                           + (f" {action.text!r}" if action.text else ""))
        row.update(chosen=chosen, kind=action.kind if action else None)
        ep.steps.append(row)
        if trace:
            trace.record(row)
        if stop or getattr(env, "terminated", False):
            ep.stop_reason = stop or "terminated"
            break
    return ep
