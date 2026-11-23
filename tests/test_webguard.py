import pyarrow as pa
import pyarrow.parquet as pq

from surety.eval.webguard import candidates_table, load_actions, missed, recall_table, target_text
from surety.gate import CriticalDetector

UUID = "7dec409b-4324-40b9-8899-74df0a2693bd"


def test_target_text_strips_prefix_uuid_and_attributes():
    assert target_text(f"annot_HIGH_Tgt_Send_request_{UUID}") == "Send request"
    assert target_text(f"annot_LOW_Tgt_name__e__value__Unsubscribe_(A_{UUID}") == "e Unsubscribe (A"
    assert target_text(f"annot_HIGH_Tgt_parent_node__[_Place_order_]_aria-label_{UUID}") == "Place order"
    assert target_text("no_match") is None


def _parquet(tmp_path, rows):
    path = tmp_path / "wg.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def _row(text, review, url="https://shop.test/p", first="LOW"):
    return {"url": url, "tagHead": "button", "annotation_folder": f"annot_{first}_Tgt_{text}_{UUID}",
            "Your Review": review, "website": "shop"}


def test_load_uses_reviewed_label_and_drops_bug_rows(tmp_path):
    path = _parquet(tmp_path, [_row("Pay_now", "HIGH", first="LOW"), _row("Home", "SAFE", first="HIGH"),
                               _row("Like", "BUG"), _row("Save", None), _row("Save", "low")])
    assert [(a["label"], a["level"]) for a in load_actions(path)] == [("Pay now", "HIGH"), ("Home", "SAFE"),
                                                                      ("Save", "LOW")]


def test_recall_table_and_misses(tmp_path):
    rows = [_row("Pay_now", "HIGH"), _row("Enviar", "HIGH"), _row("Enviar", "HIGH"), _row("Delete", "LOW"),
            _row("Home", "SAFE", url="https://shop.test/checkout")]
    actions = load_actions(_parquet(tmp_path, rows))
    table = recall_table(actions, CriticalDetector())
    assert "| label | HIGH | 3 | 0.333 [0.061, 0.792] | 1 | - |" in table
    assert "| label | SAFE | 1 | 0.000 [0.000, 0.793] | 0 | - |" in table
    assert "| label+url | SAFE | 1 | 1.000 [0.207, 1.000] | 1 | - |" in table
    assert "| label | LOW | 1 | 1.000 [0.207, 1.000] | 0 | destructive 1 |" in table
    assert missed(actions, CriticalDetector()) == [("enviar", 2)]
    cand = candidates_table(actions, CriticalDetector(), {"submit": r"submit\w*|enviar", "save": r"save\b"})
    assert "| submit | +0.667 | +0.000 | +0.000 |" in cand and "| save | +0.000 | +0.000 | +0.000 |" in cand
    assert "| all above except confirm / continue | +0.667 |" in cand
