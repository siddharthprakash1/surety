from surety.decide.base import Backend


def average_runs(runs: list[dict[str, float]]) -> dict[str, float]:
    ids = {k for r in runs for k in r}
    mean = {k: sum(r.get(k, 0.0) for r in runs) / len(runs) for k in ids}
    z = sum(mean.values()) or 1.0
    return {k: v / z for k, v in mean.items()}


def noul_debiased(backend: Backend, state_text: str, instructions: str,
                  negated_instructions: str) -> float:
    p = backend.noul(state_text, instructions)
    p_neg = backend.noul(state_text, negated_instructions)
    return 0.5 * (p + (1 - p_neg))
