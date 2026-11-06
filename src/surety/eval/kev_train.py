import json
import random
from pathlib import Path

from surety.eval.webjev import compile_instance, draw_balanced
from surety.types import Instance

QID = {"action_prediction": "action", "element_grounding": "element"}


def build_records(instances: list[Instance], plan: dict, n_train: int, n_heldout: int, seed: int,
                  max_options: int = 64) -> tuple[list[dict], list[dict]]:
    sites = set(plan["cal_sites"]) - set(plan["held_sites"])
    excluded = set(plan["cal"]) | set(plan["test_in"]) | set(plan["test_shift"])
    by_site: dict[str, list[Instance]] = {}
    for i in instances:
        if i.site in sites and i.id not in excluded and len(i.options) <= max_options:
            by_site.setdefault(i.site, []).append(i)
    rng = random.Random(seed)
    order = sorted(by_site)
    rng.shuffle(order)
    held, count = set(), 0
    for s in order:
        if count >= n_heldout:
            break
        held.add(s)
        count += len(by_site[s])

    def draw(site_set: set[str], n: int) -> list[dict]:
        by_type: dict[str, list[Instance]] = {}
        for s in sorted(site_set):
            for i in by_site[s]:
                by_type.setdefault(i.task_type, []).append(i)
        pools = [rng.sample(sorted(p, key=lambda i: i.id), len(p)) for _, p in sorted(by_type.items())]
        return [_record(i) for i in draw_balanced(pools, n)]

    heldout = draw(held, n_heldout)
    return draw(set(by_site) - held, n_train), heldout


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
