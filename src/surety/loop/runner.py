import random
from dataclasses import asdict, dataclass
from functools import partial
from urllib.parse import urlsplit

from surety.compile import NONE_OF_THESE, compile_page
from surety.decide import Backend
from surety.env import Action, Env, EnvClosedError
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
    shadow_calls: int = 0  # evaluation-only labels (shadow_rate): not part of the system's cost


def _call(fn, state: DecisionState, ep: Episode, row: dict, arg):
    value, usage = fn(state, arg)
    ep.supervisor_calls += 1
    # an explicit NONE is an answer (no option fits), not a parse failure
    if value is None and getattr(usage, "abstained", False):
        value = NONE_OF_THESE
    row["supervisor"].append({"call": fn.__name__, **asdict(usage), "answered": value is not None})
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
        max_steps: int = 30, n_perm: int = 1, calibrate=None, audit_rate: float = 0.0,
        rng: random.Random | None = None, n_perm_large: int | None = None, large_options: int = 16,
        shadow_rate: float = 0.0, shadow_rng: random.Random | None = None,
        audit_stats: list[int] | None = None) -> Episode:
    """calibrate: probs -> probs applied before the gate (the offline temperature). Online gate updates as offline
    (analyze._stream): escalated steps carry the supervisor's answer as the label (w=1); acted steps are audited with
    probability audit_rate (w=1/r; the audit answer is acted on); other steps update with w=0 (ε drift only).
    rng draws the audits: pass one Random per stream so audit positions differ across episodes.
    n_perm_large: orderings above large_options options, as the collection the gate was fitted on (5, 3 above 16).
    shadow_rate: evaluation only — an acted, unaudited step asks the supervisor (full list) with this probability and
    records the answer as row["shadow"]; nothing is acted on or learnt from it, and it is counted in shadow_calls.
    shadow_rng draws the shadows apart from the audits, so turning shadows on leaves a run's audits unchanged.
    audit_stats: [asked, answered] audits over the caller's stream. An unanswered audit yields no label, so labels
    arrive at rate r·q, not r; answered audits then weigh 1/(r·q̂), q̂ = answered / asked, which keeps the update
    unbiased when the answer rate does not depend on the label (else, and without audit_stats, 1/r: biased low).
    A dead browser (EnvClosedError, Playwright's TargetClosedError) is raised, not recorded as a failed step."""
    ep = Episode(task, [], "max_steps", 0, 0)
    if not isinstance(policy, CriticalOverride):
        policy = CriticalOverride(policy, _DETECTOR)
    detector = policy.detector
    rng = rng if rng is not None else random.Random()
    shadow_rng = shadow_rng if shadow_rng is not None else random.Random()
    history: list[str] = []
    last, streak, nones = None, 0, 0
    for i in range(max_steps):
        page = env.observe()
        state = compile_page(page, task, history)
        labels = {o.id: o.label for o in state.options}
        roles = {e.id: e.role for e in page.elements}
        policy.set_labels(labels)
        # count page options as WebJev rows count theirs: without the added none_of_these
        n_opts = sum(o.id != NONE_OF_THESE for o in state.options)
        k = n_perm_large if n_perm_large is not None and n_opts > large_options else n_perm
        answer = backend.choice(state, k)
        ep.executor_calls += 1
        probs = calibrate(answer.probs) if calibrate else answer.probs
        d = policy.decide(probs)
        row = {"step": i, "url": _bare(page.url), "n_options": len(state.options), "probs": probs,
               "decision": d.action, "reason": d.reason, "candidate_set": d.candidate_set,
               "executor_ms": answer.latency_ms, "supervisor": []}
        if calibrate:
            row["probs_raw"] = answer.probs
        ask = partial(_call, supervisor.choose, state, ep, row)
        slot = partial(_call, supervisor.slot_value, state, ep, row)

        stop, action = None, None
        chosen, weight = d.option_id, 0.0
        if d.reason.startswith("stop:"):
            stop = row["critical"] = d.reason
        else:
            if d.action == "escalate":
                # page order, as in the H3 study: probability order would leak the executor's ranking
                cands = [o.id for o in state.options if o.id in set(d.candidate_set)] or None
                chosen = ask(cands)
                if chosen in (NONE_OF_THESE, None) and cands and set(cands) < set(labels):
                    chosen = ask(None)
                weight = 1.0
            elif audit_rate and rng.random() < audit_rate:
                audited = row["audit"] = ask(None)
                if audit_stats is not None:
                    audit_stats[0] += 1
                    audit_stats[1] += audited in labels
                if audited in labels:
                    q = audit_stats[1] / audit_stats[0] if audit_stats else 1.0
                    chosen, weight = audited, 1 / (audit_rate * q)
            elif shadow_rate and shadow_rng.random() < shadow_rate:
                label, usage = supervisor.choose(state, None)
                ep.shadow_calls += 1
                row["shadow"] = NONE_OF_THESE if label is None and getattr(usage, "abstained", False) else label
        gold = chosen if chosen in labels else None
        policy.update(probs, gold or d.option_id or "", weight if gold else 0.0)
        if not stop:
            plan = _plan(chosen, roles)
            same = streak if plan and plan == last else 0
            if same >= 3:
                stop = "stuck"
            elif same == 2:
                chosen = ask(None)
                plan = _plan(chosen, roles)
            if plan and not stop:
                label = labels.get(chosen, chosen)
                cat = detector.classify(label)
                typed = plan[0] != "click" and (detector.credential(label) or roles.get(chosen) == "password")
                if cat == "payment" or typed:
                    stop = row["critical"] = "stop:payment" if cat == "payment" else "stop:credentials"
                elif cat:
                    row["critical"] = f"critical:{cat}"
            text = slot(plan[1]) if plan and plan[0] != "click" and not stop else None
            if plan and plan[0] == "select" and text and not stop:
                # the chosen value carries the effect: "Bulk actions" -> "Delete selected"
                cat = detector.classify(f"{labels.get(chosen, chosen)} {text}")
                if cat == "payment":
                    stop = row["critical"] = "stop:payment"
                elif cat:
                    row.setdefault("critical", f"critical:{cat}")
            if not stop:
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
                if isinstance(e, EnvClosedError) or type(e).__name__ == "TargetClosedError":
                    raise  # the episode could not run: the caller records error:<Type> and retries it
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
