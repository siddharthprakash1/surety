"""Learned error score vs top probability: error ranking, 200 calibration draws, adaptive streams (CPU, ~5 min)."""
import argparse
from pathlib import Path

from surety.eval.analyze import load_collection
from surety.eval.learned import ADAPTIVE, draws_table, ranking_table, static_draws, stream_draws


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", type=Path, required=True)
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--audit-rate", type=float, default=0.2)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    rows = load_collection(a.collect)
    parts = [
        f"# Learned error score: {a.collect.stem}",
        f"- collection: `{a.collect.as_posix()}`; debiased+T probabilities (T fitted on cal)\n"
        "- learned = `LearnedRisk`: L2 logistic model of P(top option wrong) from p_top, margin, normalised entropy, "
        "log option count and top-is-terminal, cross-fitted on cal (5 folds); act iff the mean fold prediction ≤ τ\n"
        "- learned@eps: τ by conformal risk control on the out-of-fold scores (E[auto_err] ≤ eps); learned_pac@eps: "
        "τ by Learn-then-Test, binomial fixed-sequence tests at δ = 0.1 (P(auto_err > eps) ≤ 0.1); risk@eps: "
        "RiskConformal, as in the M1 tables\n"
        "- the guarantees are approximate twice over: T is fitted on the same cal rows, and out-of-fold scores come from "
        "models fitted on 4/5 of them; test_shift (held-out sites) has no guarantee at all",
        "## Error ranking on the original split (AUROC of each score for 'top option wrong')", ranking_table(rows),
        f"## Calibration draws ({a.seeds}: cal re-drawn from cal ∪ test_in, T, model and gates refit)",
        draws_table(static_draws(rows, n_seeds=a.seeds)),
        *(f"## Adaptive gates on test_shift streams, eps={eps} ({a.seeds} site orders and audit draws)\n\n"
          f"{stream_draws(rows, eps, ADAPTIVE, a.audit_rate, a.seeds)}" for eps in (0.05, 0.1)),
    ]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n\n".join(parts) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
