"""Target ε (0.02–0.20) vs realized autonomous error and automation on held-out sites, over calibration draws."""
import argparse
from pathlib import Path

from surety.eval.analyze import load_collection
from surety.eval.sweep import eps_sweep, sweep_plot, sweep_table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", type=Path, required=True)
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    records = eps_sweep(load_collection(a.collect), n_seeds=a.seeds)
    png = a.out.with_suffix(".png")
    sweep_plot(records, png)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(f"# ε sweep: {a.collect.stem}\n\n- collection: `{a.collect.as_posix()}`; {a.seeds} calibration "
                     f"draws (cal re-drawn from cal ∪ test_in, T and gates refit); split test_shift\n\n"
                     f"{sweep_table(records)}\n![eps-sweep]({png.name})\n", encoding="utf-8")


if __name__ == "__main__":
    main()
