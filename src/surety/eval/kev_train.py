import json
import random
from pathlib import Path

from surety.eval.webjev import _draw, compile_instance
from surety.types import Instance

QID = {"action_prediction": "action", "element_grounding": "element"}


def build_records(instances: list[Instance], plan: dict, n_train: int, n_heldout: int, seed: int,
                  max_options: int = 64) -> tuple[list[dict], list[dict]]:
    sites = set(plan["cal_sites"]) - set(plan["held_sites"])
    excluded = set(plan["cal"]) | set(plan["test_in"]) | set(plan["test_shift"])
    by_type: dict[str, list[Instance]] = {}
    for i in instances:
        if i.site in sites and i.id not in excluded and len(i.options) <= max_options:
            by_type.setdefault(i.task_type, []).append(i)
    rng = random.Random(seed)
    pools = [rng.sample(sorted(p, key=lambda i: i.id), len(p)) for _, p in sorted(by_type.items())]
    picked = [_record(i) for i in _draw(pools, n_train + n_heldout)]
    return picked[n_heldout:], picked[:n_heldout]


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
