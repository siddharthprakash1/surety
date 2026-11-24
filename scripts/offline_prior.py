"""Option-prior correction through the M1 pipeline: accuracy, gates over calibration draws, adaptive streams (CPU)."""
import argparse
from pathlib import Path

from surety.eval.analyze import load_collection
from surety.eval.learned import ADAPTIVE, baseline, draws_table, ranking_table, static_draws, stream_draws
from surety.eval.prior import accuracy_table, corrector, prior

STREAMS = ("arisk (γ 0.02)", "alearned (γ 0.005, κ 0.2, dr): default")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", type=Path, required=True)
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--audit-rate", type=float, default=0.2)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    rows = load_collection(a.collect)
    offsets = corrector(rows).b
    records = (static_draws(rows, n_seeds=a.seeds, probs_fn=baseline)
               + static_draws(rows, n_seeds=a.seeds, probs_fn=prior, label=" +prior"))
    makers = {k: ADAPTIVE[k] for k in STREAMS}
    parts = [
        f"# Option-prior correction: {a.collect.stem}",
        f"- collection: `{a.collect.as_posix()}`\n"
        "- prior+T: `PriorCorrector` (one log-prior offset per recurring action id, L2-penalised maximum likelihood on "
        "cal; page-element ids keep 0), then T fitted on the corrected cal rows; +prior rows refit both on every draw\n"
        "- the offsets, T and the gates all use the same cal rows, so the finite-sample statements are approximate, as "
        "in M1; test_shift (held-out sites) has no guarantee",
        "## Offsets fitted on the original cal split\n\n"
        + ", ".join(f"{o} {v:+.2f}" for o, v in sorted(offsets.items(), key=lambda kv: kv[1])),
        "## Accuracy and calibration (original split)", accuracy_table(rows),
        "## Error ranking with prior+T (AUROC for 'top option wrong')", ranking_table(rows, probs_fn=prior),
        f"## Calibration draws ({a.seeds}; offsets, T, model and gates refit per draw)", draws_table(records),
        "## The same draws by task type", draws_table(records, by_type=True),
        *(f"## Adaptive gates on test_shift streams, eps={eps}, prior+T ({a.seeds} site orders and audit draws)\n\n"
          f"{stream_draws(rows, eps, makers, a.audit_rate, a.seeds, probs_fn=prior)}" for eps in (0.05, 0.1)),
    ]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n\n".join(parts) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
