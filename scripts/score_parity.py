"""Did a re-score reproduce a collection? Agreement on the steps both files share.

    uv run python scripts/score_parity.py artifacts/m1/kev_compact.jsonl runs/scores/m1-parity/rows.jsonl
"""
import argparse
import json

from surety.eval.parity import load_rows, parity

ap = argparse.ArgumentParser()
ap.add_argument("a")
ap.add_argument("b")
a = ap.parse_args()
print(json.dumps(parity(load_rows(a.a), load_rows(a.b)), indent=1))
