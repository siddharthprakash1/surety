import json
import random
import re
from pathlib import Path
from typing import Literal

import pyarrow.parquet as pq

from surety.types import DecisionState, Instance, Option

_COLUMNS = ["id", "task_type", "question_type", "options", "gold", "context", "question", "num_tokens",
            "context_format", "website"]
_EG_OPTION = re.compile(r"((\d+)(?::\d+)?): (\{.*\})", re.S)
_LABEL_CHARS = 100
_GOAL_FIELDS = {
    "Task": "The user's original task (global constraints; no stage may drop any of its conditions): ",
    "Completed stages": "Completed stages (keep their conditions; do not repeat them): ",
    "Current subgoal": "[The only goal of the current stage] ",
    "Guidance": "Temporary semantic guidance after a stall (choose only from visible evidence; "
                "you still decide the operation and the target): ",
    "Rejected fields": "The text helper explicitly rejected these fields in the same context and typed nothing; "
                       "do not choose them for now: ",
}
_INSTRUCTIONS = {
    "operation": "Which operation should the agent take next?",
    "click_target": "Which element should the agent click?",
    "type_target": "Which element should the agent type into?",
    "select_target": "Which option should the agent select?",
}


def load_instances(root: Path,
                   task_types: tuple[str, ...] = ("action_prediction", "element_grounding")) -> list[Instance]:
    root = Path(root)
    files = sorted(p for p in root.rglob("*.parquet") if not any(s.startswith(".") for s in p.relative_to(root).parts))
    rows = [r for f in files
            for r in pq.read_table(f, columns=_COLUMNS, filters=[("task_type", "in", list(task_types))]).to_pylist()]
    return [Instance(site=r.pop("website"), **r) for r in rows]


def site_split(instances, holdout_frac: float = 0.3, seed: int = 0) -> tuple[set[str], set[str]]:
    sites = sorted({i.site for i in instances})
    random.Random(seed).shuffle(sites)
    n_held = round(len(sites) * holdout_frac)
    return set(sites[n_held:]), set(sites[:n_held])


def make_plan(instances, cal_sites, held_sites, n_cal: int, n_test_in: int, n_test_shift: int,
              seed: int = 0) -> dict[str, list[str]]:
    rng = random.Random(seed)

    def pools(sites):
        by_type: dict[str, list[str]] = {}
        for i in instances:
            if i.site in sites:
                by_type.setdefault(i.task_type, []).append(i.id)
        return [rng.sample(sorted(ids), len(ids)) for _, ids in sorted(by_type.items())]

    cal_pools = pools(cal_sites)
    return {"cal": _draw(cal_pools, n_cal), "test_in": _draw(cal_pools, n_test_in),
            "test_shift": _draw(pools(held_sites), n_test_shift)}


def _draw(pools: list[list[str]], n: int) -> list[str]:
    out: list[str] = []
    while len(out) < n and any(pools):
        for p in pools:
            if p and len(out) < n:
                out.append(p.pop())
    return out


def option_id_and_label(option: str, task_type: str) -> tuple[str, str]:
    if task_type == "element_grounding" and (m := _EG_OPTION.fullmatch(option)):
        j = json.loads(m.group(3))
        label = re.sub(r"^\[[^\]]*\]\s*", "", j["element"])
        flags = [(k, v) for k, v in j.items() if k not in ("element", "current_value", "role")]
        return "E" + m.group(1), _summary(j.get("role", ""), label, j.get("current_value", ""), flags)
    head, _, rest = option.partition(": ")
    return head, rest


def compile_instance(inst: Instance, mode: Literal["compact", "full"] = "compact", max_page_chars: int = 1500,
                     max_elements: int = 40) -> DecisionState:
    pairs = [option_id_and_label(o, inst.task_type) for o in inst.options]
    ids = _dedupe([i for i, _ in pairs])
    if mode == "compact" and inst.context_format == "structured":
        state_text = _compact(inst, max_page_chars, max_elements)
    else:
        state_text = inst.context + "\n\n" + inst.question
    return DecisionState(
        state_text=state_text,
        instructions=_INSTRUCTIONS.get(inst.question_type, "Which element should the agent act on?"),
        options=[Option(i, label) for i, (_, label) in zip(ids, pairs)],
        meta={"gold_id": ids[inst.gold]},
    )


def _dedupe(ids: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for i in ids:
        new, k = i, 1
        while new in seen:
            k += 1
            new = f"{i}_{k}"
        seen.add(new)
        out.append(new)
    return out


def _compact(inst: Instance, max_page_chars: int, max_elements: int) -> str:
    ctx, q = json.loads(inst.context), json.loads(inst.question)
    page = ctx["page"]
    referenced = {m.group(2) for o in inst.options if (m := _EG_OPTION.fullmatch(o))}
    lines = _goal_lines(q["goal"])
    if "operation" in q:
        lines.append(f"Operation: {q['operation']}")
    lines += [f"URL: {page['url']}", f"Title: {page['title']}", "Recent actions:"]
    lines += [_action_line(a) for a in ctx["recent_actions"]] or ["(none)"]
    lines += ["Page text:", page["text"][:max_page_chars], "Elements:"]
    lines += [_element_line(e) for e in _shortlist(ctx["elements"], referenced, max_elements)]
    return "\n".join(lines)


def _goal_lines(goal: str) -> list[str]:
    lines = goal.strip().splitlines()
    out = [line for line in lines if line.startswith("Today is ")]
    for name, prefix in _GOAL_FIELDS.items():
        out += [f"{name}: {line.removeprefix(prefix)}" for line in lines if line.startswith(prefix)]
    return out


def _action_line(a: dict) -> str:
    text = f' "{a["text"]}"' if a.get("text") else ""
    return f"- {a['kind']}: {a['action']}{text}" + (" (page changed)" if a.get("page_changed") else "")


def _shortlist(elements: list[dict], referenced: set[str], cap: int) -> list[dict]:
    room = max(0, cap - sum(e["index"] in referenced for e in elements))
    keep = []
    for e in elements:
        if e["index"] in referenced:
            keep.append(e)
        elif room:
            keep.append(e)
            room -= 1
    return keep


def _element_line(e: dict) -> str:
    flags = [(k, e[k]) for k in ("expanded", "checked", "selected") if k in e]
    summary = _summary(e.get("role", ""), e["label"][:_LABEL_CHARS], e["value"][:_LABEL_CHARS], flags)
    return f"[{e['index']}] {summary} | ops={','.join(e['operations'])}"


def _summary(role: str, label: str, value: str, flags: list[tuple[str, str]]) -> str:
    parts = [role, label, f"value={value}" if value else "", *(f"{k}={v}" for k, v in flags)]
    return " | ".join(p for p in parts if p)
