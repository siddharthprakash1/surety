import json

import numpy as np
import pytest

from surety.calib import metrics as m
from surety.eval.analyze import (
    SPLITS,
    ablation_table,
    aci_simulation,
    h1_table,
    h2_table,
    load_collection,
    risk_coverage_plot,
    variants,
)

K = 5


def synthetic_rows(n_per_split, seed, shift_sharpen=1.0, k=K, n_sites=20):
    rng = np.random.default_rng(seed)
    rows = []
    for split in SPLITS:
        for i in range(n_per_split):
            p = np.exp(rng.normal(size=k) * 2)
            p /= p.sum()
            gold = rng.choice(k, p=p)
            if split == "test_shift":
                p = p**shift_sharpen / (p**shift_sharpen).sum()
            first = p * np.array([3] + [1] * (k - 1))  # position bias toward the first-listed option
            first /= first.sum()
            probs = {f"o{j}": float(v) for j, v in enumerate(p)}
            runs = [{f"o{j}": float(v) for j, v in enumerate(first)}, probs]
            site = f"{'held' if split == 'test_shift' else 'seen'}{i % n_sites}"
            rows.append({"id": f"{split}-{i}", "split": split, "site": site, "gold_id": f"o{gold}",
                         "option_ids": list(probs), "probs": probs, "runs": runs})
    return rows


@pytest.fixture(scope="module")
def shifted():
    return synthetic_rows(2000, 0, shift_sharpen=3.0)


def records(table):
    header, _, *body = [[c.strip() for c in line.strip().strip("|").split("|")]
                        for line in table.strip().splitlines()]
    return [dict(zip(header, row)) for row in body]


def row(recs, **keys):
    (match,) = [r for r in recs if all(r[k] == v for k, v in keys.items())]
    return match


def values(rec, *cols):
    return tuple(float(rec[c]) for c in cols)


def test_load_collection_reads_jsonl(tmp_path):
    rows = synthetic_rows(3, 0)
    path = tmp_path / "c.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n" + json.dumps(rows[0])[:20])
    assert load_collection(path) == rows


def test_load_collection_keeps_unterminated_complete_line(tmp_path):
    rows = synthetic_rows(3, 0)
    path = tmp_path / "c.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    assert load_collection(path) == rows


def test_variants_requires_cal_split():
    with pytest.raises(ValueError, match="no cal split"):
        variants([r for r in synthetic_rows(3, 0) if r["split"] != "cal"])


def test_h1_table_has_all_splits(shifted):
    recs = records(h1_table(shifted))
    assert [(r["split"], r["variant"]) for r in recs] == [
        (s, v) for s in SPLITS for v in ("first_run", "debiased", "debiased+T")]
    pins = {("test_in", "first_run"): (0.602, 0.053), ("test_in", "debiased"): (0.634, 0.025),
            ("test_shift", "debiased+T"): (0.643, 0.221)}
    for (split, variant), pin in pins.items():
        assert values(row(recs, split=split, variant=variant), "acc", "ECE") == pytest.approx(pin, abs=1e-3)


def test_h2_conformal_autonomous_error_below_eps_in_dist(shifted):
    recs = records(h2_table(shifted))
    for eps in (0.05, 0.1):
        for policy in ("conformal", "risk"):
            assert float(row(recs, eps=str(eps), policy=f"{policy}@{eps}", split="test_in")["auto_err"]) <= eps


def test_h2_pinned_rows(shifted):
    recs = records(h2_table(shifted, eps_list=(0.1,)))
    pins = {
        ("fitted@0.1", "test_in"): (0.094, 0.471, 0.906, 3.114),
        ("fitted@0.1", "test_shift"): (0.275, 0.864, 0.725, 1.544),
        ("conformal@0.1", "test_in"): (0.024, 0.240, 0.909, 2.178),
        ("conformal@0.1", "test_shift"): (0.168, 0.649, 0.777, 1.409),
        ("risk@0.1", "test_in"): (0.086, 0.455, 0.831, 1.635),
        ("risk@0.1", "test_shift"): (0.219, 0.765, 0.739, 1.249),
    }
    for (policy, split), pin in pins.items():
        got = values(row(recs, policy=policy, split=split), "auto_err", "automation", "coverage", "set_size")
        assert got == pytest.approx(pin, abs=1e-3)


def test_auto_err_ci_brackets_rate(shifted):
    recs = records(h2_table(shifted, eps_list=(0.1,))) + records(aci_simulation(shifted, 0.1))
    for r in recs:
        lo, hi = (float(x) for x in r["auto_err_95ci"].strip("[]").split(","))
        assert lo <= float(r["auto_err"]) <= hi and hi - lo > 0


def test_h2_threshold_sets_are_singleton_or_full(shifted):
    recs = [r for r in records(h2_table(shifted, eps_list=(0.1,)))
            if r["policy"].split("@")[0] in ("fixed", "margin", "fitted", "per_choice")]
    assert len(recs) == 10
    for r in recs:
        auto_err, automation, coverage, set_size = values(r, "auto_err", "automation", "coverage", "set_size")
        assert coverage == pytest.approx(1 - auto_err, abs=2e-3)
        assert set_size == pytest.approx(automation + (1 - automation) * K, abs=3e-3)


def test_aci_restores_target_under_shift(shifted):
    eps = 0.1
    recs = records(aci_simulation(shifted, eps))
    cols = ("auto_err", "tail_auto_err", "automation", "final_alpha")
    pins = {("conformal", "aci"): (0.062, 0.040, 0.303, 0.020), ("risk", "arisk"): (0.106, 0.125, 0.414, 0.001)}
    for (static, adaptive), pin in pins.items():
        got = row(recs, policy=f"{adaptive}@{eps}", audit_rate="0.2")
        assert float(got["auto_err"]) <= 1.5 * eps < float(row(recs, policy=f"{static}@{eps}")["auto_err"])
        assert values(got, *cols) == pytest.approx(pin, abs=1e-3)
    assert float(row(recs, policy=f"aci@{eps}", audit_rate="0")["final_alpha"]) == 0.5


def test_ablation_table_joins_by_id():
    compact = [{**r, "mode": "compact", "latency_ms": 100.0, "state_chars": 1000, "n_perm": 5}
               for r in synthetic_rows(30, 0) if r["split"] == "test_in"]
    shared = compact[::3]
    full = [{**r, "mode": "full", "latency_ms": 300.0, "state_chars": 8000, "n_perm": 3,
             "probs": {o: float(o == r["gold_id"]) for o in r["option_ids"]}} for r in shared]
    recs = records(ablation_table(compact, full + [{**full[0], "id": "only-full"}]))
    acc = m.accuracy([r["probs"] for r in shared], [r["gold_id"] for r in shared])
    assert [r["mode"] for r in recs] == ["compact", "full"]
    for rec, pin in zip(recs, [(10, acc, 100, 1000, 5), (10, 1.0, 300, 8000, 3)]):
        assert values(rec, "n", "acc", "latency_ms", "state_chars", "n_perm") == pytest.approx(pin, abs=1e-3)


def test_risk_coverage_plot_writes_png(tmp_path):
    out = tmp_path / "rc.png"
    risk_coverage_plot(synthetic_rows(100, 0), out)
    assert out.read_bytes()[:4] == b"\x89PNG"
