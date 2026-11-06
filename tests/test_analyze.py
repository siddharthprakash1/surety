import json

import numpy as np
import pytest

from surety.eval.analyze import (
    SPLITS,
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
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
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
        ("conformal@0.1", "test_in"): (0.024, 0.239, 0.909, 2.179),
        ("conformal@0.1", "test_shift"): (0.168, 0.649, 0.777, 1.409),
        ("risk@0.1", "test_in"): (0.086, 0.455, 0.831, 1.635),
        ("risk@0.1", "test_shift"): (0.219, 0.765, 0.739, 1.249),
    }
    for (policy, split), pin in pins.items():
        got = values(row(recs, policy=policy, split=split), "auto_err", "automation", "coverage", "set_size")
        assert got == pytest.approx(pin, abs=1e-3)


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
    static = float(row(recs, policy=f"conformal@{eps}")["auto_err"])
    aci = row(recs, policy=f"aci@{eps}", audit_rate="0.2")
    assert float(aci["auto_err"]) <= 1.5 * eps < static
    cols = ("auto_err", "tail_auto_err", "automation", "final_alpha")
    assert values(aci, *cols) == pytest.approx((0.095, 0.095, 0.466, 0.041), abs=1e-3)
    assert values(row(recs, policy=f"arisk@{eps}", audit_rate="0.2"), *cols) == pytest.approx(
        (0.177, 0.200, 0.661, 0.223), abs=1e-3)
    assert float(row(recs, policy=f"aci@{eps}", audit_rate="0")["final_alpha"]) == 0.5


def test_risk_coverage_plot_writes_png(tmp_path):
    out = tmp_path / "rc.png"
    risk_coverage_plot(synthetic_rows(100, 0), out)
    assert out.read_bytes()[:4] == b"\x89PNG"
