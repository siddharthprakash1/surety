from dataclasses import asdict, dataclass
from functools import partial
from urllib.parse import urlsplit

from surety.compile import NONE_OF_THESE, compile_page
from surety.decide import Backend
from surety.env import Action, Env
from surety.gate import CriticalDetector, CriticalOverride, GatePolicy
from surety.trace import Trace
from surety.types import DecisionState

TYPE_ROLES = {"input", "textbox", "textarea", "searchbox", "spinbutton", "password"}
SELECT_ROLES = {"select", "combobox"}
_DETECTOR = CriticalDetector()


@dataclass
class Episode:
    task: str
    steps: list[dict]
    stop_reason: str
    supervisor_calls: int
    executor_calls: int


def _call(fn, state: DecisionState, ep: Episode, row: dict, arg):
    value, usage = fn(state, arg)
    ep.supervisor_calls += 1
    row["supervisor"].append({"call": fn.__name__, **asdict(usage)})
    return value


def _plan(option_id: str | None, roles: dict[str, str]) -> tuple[str, str] | None:
    if option_id in (NONE_OF_THESE, None):
        return None
    role = roles.get(option_id)
    return ("type" if role in TYPE_ROLES else "select" if role in SELECT_ROLES else "click"), option_id


def _bare(url: str) -> str:
    u = urlsplit(url)
    return u._replace(netloc=u.netloc.rpartition("@")[2], query="", fragment="").geturl()


def run(task: str, env: Env, backend: Backend, policy: GatePolicy, supervisor, trace: Trace | None = None,
        max_steps: int = 30, n_perm: int = 1) -> Episode:
    ep = Episode(task, [], "max_steps", 0, 0)
    if not isinstance(policy, CriticalOverride):
        policy = CriticalOverride(policy, _DETECTOR)
    detector = policy.detector
    history: list[str] = []
    last, streak, nones = None, 0, 0
    for i in range(max_steps):
        page = env.observe()
        state = compile_page(page, task, history)
        labels = {o.id: o.label for o in state.options}
        roles = {e.id: e.role for e in page.elements}
        policy.set_labels(labels)
        answer = backend.choice(state, n_perm)
        ep.executor_calls += 1
        d = policy.decide(answer.probs)
        row = {"step": i, "url": _bare(page.url), "n_options": len(state.options), "probs": answer.probs,
               "decision": d.action, "reason": d.reason, "candidate_set": d.candidate_set,
               "executor_ms": answer.latency_ms, "supervisor": []}
        ask = partial(_call, supervisor.choose, state, ep, row)
        slot = partial(_call, supervisor.slot_value, state, ep, row)

        stop, action = None, None
        chosen = d.option_id
        if d.reason.startswith("stop:"):
            stop = d.reason
        else:
            if d.action == "escalate":
                cands = d.candidate_set or None
                chosen = ask(cands)
                if chosen in (NONE_OF_THESE, None) and cands and set(cands) < set(labels):
                    chosen = ask(None)
            plan = _plan(chosen, roles)
            same = streak if plan and plan == last else 0
            if same >= 3:
                stop = "stuck"
            elif same == 2:
                chosen = ask(None)
                plan = _plan(chosen, roles)
            if plan and not stop:
                cat = detector.classify(labels.get(chosen, chosen))
                typed = plan[0] != "click" and (cat == "credentials" or roles.get(chosen) == "password")
                if cat == "payment" or typed:
                    stop = row["critical"] = "stop:payment" if cat == "payment" else "stop:credentials"
                elif cat:
                    row["critical"] = f"critical:{cat}"
            if not stop:
                text = slot(plan[1]) if plan and plan[0] != "click" else None
                if plan and (plan[0] == "click" or text):
                    action = Action(plan[0], plan[1], text)
                    nones = 0
                else:
                    # a missing slot value counts as none_of_these; an unanswered escalation does not
                    nones = nones + 1 if chosen == NONE_OF_THESE or plan else 0
                    stop = "done" if nones == 2 else None
                    action = Action("done" if stop else "scroll")

        if action:
            try:
                env.act(action)
            except Exception as e:
                # only the type: env messages can echo typed text
                row["error"] = type(e).__name__
                stop = "error"
            else:
                streak = streak + 1 if (action.kind, action.element_id) == last else 1
                last = action.kind, action.element_id
                history.append(f"{action.kind} {labels.get(action.element_id, '')}".strip()
                               + (f" {action.text!r}" if action.text else ""))
        row.update(chosen=chosen, kind=action.kind if action else None)
        ep.steps.append(row)
        if trace:
            trace.record(row)
        stop = stop or next((f for f in ("terminated", "truncated") if getattr(env, f, False)), None)
        if stop:
            ep.stop_reason = stop
            break
    return ep
