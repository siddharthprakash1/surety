import argparse
from collections import Counter
from pathlib import Path

from surety.eval.analyze import SPLITS, aci_simulation, h1_table, h2_table, load_collection, risk_coverage_plot

EPS = (0.05, 0.1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = load_collection(args.collect)
    counts = Counter(r["split"] for r in rows)
    png = args.out.with_suffix(".png")
    risk_coverage_plot(rows, png)
    parts = [
        f"# M1 analysis: {args.collect.stem}",
        f"- collection: `{args.collect}`\n"
        f"- rows: {', '.join(f'{s} {counts[s]}' for s in SPLITS)}\n"
        f"- backend: {', '.join(sorted({r['backend'] for r in rows}))}\n"
        f"- mode: {', '.join(sorted({r['mode'] for r in rows}))}",
        "## H1: calibration", h1_table(rows),
        "## H2: gates fitted on cal (debiased+T)", h2_table(rows, EPS),
        *(f"## H2: ACI on test_shift, eps={eps}\n\n{aci_simulation(rows, eps)}" for eps in EPS),
        f"![risk-coverage]({png.name})",
    ]
    args.out.write_text("\n\n".join(parts) + "\n")


if __name__ == "__main__":
    main()
