"""Marginal vs per-task-type risk control on calibration sites (test_in) and held-out sites (test_shift)."""
import argparse
from pathlib import Path

from surety.eval.analyze import load_collection
from surety.eval.groups import adaptive_group_streams, group_draws, group_table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", type=Path, required=True)
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    rows = load_collection(a.collect)
    tables = [f"## {split}\n\n{group_table(group_draws(rows, n_seeds=a.seeds, split=split))}"
              for split in ("test_in", "test_shift")]
    tables += [f"## Adaptive risk on test_shift streams, eps={eps}, audit rate 0.2\n\n"
               f"{adaptive_group_streams(rows, eps, n_seeds=a.seeds)}" for eps in (0.05, 0.1)]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(f"# Group-conditional risk control: {a.collect.stem}\n\n- collection: `{a.collect.as_posix()}`; "
                     f"{a.seeds} calibration draws (cal re-drawn from cal ∪ test_in, T pooled and refit); per_group = "
                     f"one RiskConformal per task type on that type's cal rows; test_in = calibration sites, "
                     f"test_shift = held-out sites\n\n" + "\n\n".join(tables), encoding="utf-8")


if __name__ == "__main__":
    main()
