"""Agreement of two collections on the steps they share, e.g. M1's rows and a Modal re-score of the same plan."""
import json
from pathlib import Path
from statistics import fmean


def load_rows(path) -> dict[str, dict]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return {r["id"]: r for r in (json.loads(line) for line in lines if line.strip())}


def _top(r: dict) -> str:
    return max(r["probs"], key=r["probs"].get)


def parity(a: dict[str, dict], b: dict[str, dict]) -> dict:
    """Shared steps, same option lists, top-1 agreement, accuracy of each, and |p_a - p_b| over a's options."""
    shared = sorted(set(a) & set(b))
    out = {"shared": len(shared), "only_a": len(set(a) - set(b)), "only_b": len(set(b) - set(a))}
    if not shared:
        return out
    gaps = [abs(p - b[i]["probs"].get(o, 0.0)) for i in shared for o, p in a[i]["probs"].items()]
    return out | {"same_options": fmean(a[i]["option_ids"] == b[i]["option_ids"] for i in shared),
                  "top1_agree": fmean(_top(a[i]) == _top(b[i]) for i in shared),
                  "acc_a": fmean(_top(a[i]) == a[i]["gold_id"] for i in shared),
                  "acc_b": fmean(_top(b[i]) == b[i]["gold_id"] for i in shared),
                  "mean_abs_diff": fmean(gaps), "max_abs_diff": max(gaps)}
