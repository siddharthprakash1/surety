"""Option-prior correction of the executor (PriorCorrector) through the M1 pipeline (results/m1/*_prior.md).

prior: the debiased probabilities corrected by per-action log-prior offsets fitted on the rows' cal split, then
temperature-scaled with T fitted on the corrected cal rows. Gates are fitted on the same cal rows, as in M1.
"""
from surety.calib import metrics as m
from surety.decide import PriorCorrector, TemperatureScaler
from surety.eval.analyze import _split, _table
from surety.eval.learned import baseline

SPLITS = ("cal", "test_in", "test_shift")


def corrector(rows: list[dict]) -> PriorCorrector:
    pc = PriorCorrector()
    pc.fit(*_split(rows, [r["probs"] for r in rows], "cal"))
    return pc


def prior(rows: list[dict]) -> list[dict[str, float]]:
    pc = corrector(rows)
    corrected = [pc.apply(r["probs"]) for r in rows]
    scaler = TemperatureScaler()
    scaler.fit(*_split(rows, corrected, "cal"))
    return [scaler.apply(p) for p in corrected]


def accuracy_table(rows: list[dict]) -> str:
    """Top-1 accuracy and ECE, M1 probabilities vs prior-corrected, by split and task type."""
    pipelines = {"debiased+T": baseline(rows), "prior+T": prior(rows)}
    body = []
    for split in SPLITS:
        for task in ("all", "action_prediction", "element_grounding"):
            idx = [i for i, r in enumerate(rows) if r["split"] == split and task in ("all", r["task_type"])]
            g = [rows[i]["gold_id"] for i in idx]
            row = [split, task, len(idx)]
            for probs in pipelines.values():
                p = [probs[i] for i in idx]
                row += [m.accuracy(p, g), m.ece(p, g)]
            body.append(row)
    return _table(("split", "task_type", "n", "acc debiased+T", "ECE debiased+T", "acc prior+T", "ECE prior+T"), body)
