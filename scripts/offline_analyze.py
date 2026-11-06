import argparse
from collections import Counter
from pathlib import Path

from surety.eval.analyze import (
    SPLITS,
    TAIL,
    aci_simulation,
    h1_table,
    h2_table,
    h3_table,
    load_collection,
    risk_coverage_plot,
    variants,
)

EPS = (0.05, 0.1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", type=Path, required=True)
    ap.add_argument("--h3", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = load_collection(args.collect)
    vs = variants(rows)
    counts = Counter(r["split"] for r in rows)
    png = args.out.with_suffix(".png")
    risk_coverage_plot(rows, png, vs=vs)
    parts = [
        f"# M1 analysis: {args.collect.stem}",
        f"- collection: `{args.collect}`\n"
        f"- rows: {', '.join(f'{s} {counts[s]}' for s in SPLITS)}\n"
        f"- backend: {', '.join(sorted({r['backend'] for r in rows}))}\n"
        f"- mode: {', '.join(sorted({r['mode'] for r in rows}))}",
        "## H1: calibration", h1_table(rows, vs=vs),
        "## H2: gates fitted on cal (debiased+T)", h2_table(rows, EPS, vs=vs),
        *(f"## H2: adaptive gates on test_shift, eps={eps} (tail = last {TAIL} steps)\n\n"
          f"{aci_simulation(rows, eps, vs=vs)}" for eps in EPS),
        f"![risk-coverage]({png.name})",
    ]
    if args.h3:
        parts += ["## H3: supervisor on full list vs conformal set", h3_table(load_collection(args.h3))]
    args.out.write_text("\n\n".join(parts) + "\n")


if __name__ == "__main__":
    main()
