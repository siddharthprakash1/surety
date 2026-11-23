"""Calibration-draw spread, site-cluster bootstrap CIs and adaptive-stream spread for an offline collection."""
import argparse
from pathlib import Path

from surety.eval.analyze import load_collection
from surety.eval.resample import adaptive_draws, draws, draws_table, site_bootstrap

EPS = (0.05, 0.1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", type=Path, required=True)
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    rows = load_collection(a.collect)
    parts = [
        f"# Uncertainty: {a.collect.stem}",
        f"- collection: `{a.collect.as_posix()}`; {a.seeds} calibration draws, "
        f"{a.boot} site-bootstrap resamples, {a.seeds} adaptive streams",
        f"## Calibration draws: cal re-drawn from cal ∪ test_in ({a.seeds} draws), T and gates refit, debiased+T",
        draws_table(draws(rows, EPS, a.seeds)),
        "## Site-cluster bootstrap: original cal fit, test sites resampled with replacement",
        site_bootstrap(rows, EPS, a.boot),
        *(f"## Adaptive risk on test_shift, eps={eps}: site order and audits redrawn per stream\n\n"
          f"{adaptive_draws(rows, eps, n_seeds=a.seeds)}" for eps in EPS),
    ]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n\n".join(parts) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
