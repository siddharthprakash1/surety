import json
import random
from pathlib import Path

from surety.eval.webjev import compile_instance, draw_balanced
from surety.types import Instance

QID = {"action_prediction": "action", "element_grounding": "element"}


def build_records(instances: list[Instance], plan: dict, n_train: int, n_heldout: int, seed: int,
                  max_options: int = 64, heldout_site_cap: int = 60) -> tuple[list[dict], list[dict]]:
    # a spread (M2) plan names its training sites; an M1 plan trains on its calibration sites, minus the plan's own steps
    sites = set(plan["train_sites"]) if "train_sites" in plan else set(plan["cal_sites"]) - set(plan["held_sites"])
    excluded = set(plan["cal"]) | set(plan["test_in"]) | set(plan["test_shift"])
    by_site: dict[str, list[Instance]] = {}
    for i in instances:
        if i.site in sites and i.id not in excluded and len(i.options) <= max_options:
            by_site.setdefault(i.site, []).append(i)
    rng = random.Random(seed)
    order = sorted(by_site)
    rng.shuffle(order)
    held, count = [], 0
    for s in order:
        if count >= n_heldout:
            break
        held.append(s)
        count += min(len(by_site[s]), heldout_site_cap)

    def pools(insts: list[Instance]) -> list[list[Instance]]:
        by_type: dict[str, list[Instance]] = {}
        for i in sorted(insts, key=lambda i: i.id):
            by_type.setdefault(i.task_type, []).append(i)
        return [rng.sample(p, len(p)) for _, p in sorted(by_type.items())]

    capped = [i for s in held for i in draw_balanced(pools(by_site[s]), heldout_site_cap)]
    heldout = draw_balanced(pools(capped), n_heldout)
    rest = set(by_site) - set(held)
    train = draw_balanced(pools([i for s in rest for i in by_site[s]]), n_train)
    return [_record(i) for i in train], [_record(i) for i in heldout]


def _record(inst: Instance) -> dict:
    state = compile_instance(inst, "compact")
    question = {"type": "choice", "instructions": state.instructions,
                "criteria": {o.id: o.label for o in state.options}, "label": state.meta["gold_id"], "src": "webjev"}
    # kev.data.load_records keeps a caller's _meta (never shown to the model); id makes benchmark rows joinable
    return {"state": state.state_text, "questions": {QID[inst.task_type]: question}, "_meta": {"id": inst.id}}


def write_jsonl(records: list[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
