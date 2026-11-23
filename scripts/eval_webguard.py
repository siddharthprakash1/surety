"""Critical-action detector vs WebGuard risk labels (spec §Safety).

Data: uvx --from huggingface_hub hf download boyuanzheng010/webguard --repo-type dataset --local-dir data/webguard_alt/webguard
(CC-BY-NC-4.0 via osunlp/WebGuard: evaluation only, not redistributed.)
"""
import argparse
from collections import Counter
from pathlib import Path

from surety.eval.webguard import LEVELS, candidates_table, load_actions, missed, recall_table
from surety.gate import CriticalDetector


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/webguard_alt/webguard/data/train-00000-of-00001.parquet"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    actions = load_actions(a.data)
    det = CriticalDetector()
    counts = Counter(x["level"] for x in actions)
    misses = missed(actions, det)
    parts = [
        "# WebGuard: critical-action detector recall",
        f"- actions: {len(actions)} ({', '.join(f'{lv} {counts[lv]}' for lv in reversed(LEVELS))}); "
        f"sites: {len({x['site'] for x in actions})}",
        "- label: target text from the annotation export (≤30 chars); url: the page URL",
        "- flagged = any category (payment stops; the rest escalate to the supervisor); CI = Wilson 95%",
        recall_table(actions, det),
        "## Most frequent HIGH actions the detector misses (label+url)",
        "| label | n |\n|---|---|\n" + "\n".join(f"| {t} | {n} |" for t, n in misses),
        "## Candidate stem families (not in the detector): share of each level they would newly flag (label+url)",
        "Chosen after reading the misses above, so the +HIGH column is in-sample; flagged non-payment actions "
        "escalate to the supervisor (cost: supervisor calls), they do not stop the episode.",
        candidates_table(actions, det),
    ]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n\n".join(parts) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
