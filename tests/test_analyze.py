import json

import numpy as np

from surety.eval.analyze import (
    aci_simulation,
    h1_table,
    h2_table,
    load_collection,
    risk_coverage_plot,
)

SPLITS = ("cal", "test_in", "test_shift")


def synthetic_rows(n_per_split, seed, shift_sharpen=1.0, k=5, n_sites=20):
    rng = np.random.default_rng(seed)
    rows = []
    for split in SPLITS:
        for i in range(n_per_split):
            p = np.exp(rng.normal(size=k) * 2)
            p /= p.sum()
            gold = rng.choice(k, p=p)
            if split == "test_shift":
                p = p**shift_sharpen / (p**shift_sharpen).sum()
            probs = {f"o{j}": float(v) for j, v in enumerate(p)}
            site = f"{'held' if split == 'test_shift' else 'seen'}{i % n_sites}"
            rows.append({"id": f"{split}-{i}", "split": split, "site": site, "gold_id": f"o{gold}",
                         "option_ids": list(probs), "probs": probs, "runs": [probs]})
    return rows


def records(table):
    header, _, *body = [[c.strip() for c in line.strip().strip("|").split("|")]
                        for line in table.strip().splitlines()]
    return [dict(zip(header, row)) for row in body]


def row(recs, **keys):
    (match,) = [r for r in recs if all(r[k] == v for k, v in keys.items())]
    return match


def test_load_collection_reads_jsonl(tmp_path):
    rows = synthetic_rows(3, 0)
    path = tmp_path / "c.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert load_collection(path) == rows


def test_h1_table_has_all_splits():
    recs = records(h1_table(synthetic_rows(200, 0)))
    variants = ("first_run", "debiased", "debiased+T")
    assert [(r["split"], r["variant"]) for r in recs] == [(s, v) for s in SPLITS for v in variants]


def test_h2_conformal_autonomous_error_below_eps_in_dist():
    recs = records(h2_table(synthetic_rows(2000, 0)))
    for eps in (0.05, 0.1):
        assert float(row(recs, eps=str(eps), policy=f"conformal@{eps}", split="test_in")["auto_err"]) <= eps


def test_aci_restores_target_under_shift():
    eps = 0.1
    recs = records(aci_simulation(synthetic_rows(2000, 0, shift_sharpen=3.0), eps))
    static = float(row(recs, policy=f"conformal@{eps}")["auto_err"])
    aci = float(row(recs, policy=f"aci@{eps}", audit_rate="0.2")["auto_err"])
    assert aci <= 1.5 * eps < static


def test_risk_coverage_plot_writes_png(tmp_path):
    out = tmp_path / "rc.png"
    risk_coverage_plot(synthetic_rows(100, 0), out)
    assert out.read_bytes()[:4] == b"\x89PNG"
