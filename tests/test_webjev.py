import json
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from surety.eval.webjev import (
    compile_instance,
    load_instances,
    make_plan,
    option_id_and_label,
    site_split,
)
from surety.types import Instance

AP, EG = "action_prediction", "element_grounding"
AP_OPTIONS = [
    "CLICK: Click an element.",
    "TYPE_TEXT: Enter text.",
    "E72: Scroll down within the region: Edition / US",
    "DONE: Every requirement is visibly satisfied.",
]


def _question(stage, operation=None):
    goal = (
        "\nToday is 2026-09-24; tomorrow is 2026-09-25.\n"
        "The user's original task (global constraints; no stage may drop any of its conditions): Buy socks.\n"
        "In the current stage complete only this stage's goal; later stages handle the rest.\n"
        f"[The only goal of the current stage] {stage}\n"
        "Choose DONE as soon as this stage's goal above is met."
    )
    q = {"goal": goal, "rules": "Advance the user's goal from the CURRENT page. " * 40}
    return json.dumps(q | ({"operation": operation} if operation else {}))


def _context(n_elements):
    elements = [{"role": "button", "value": "", "index": str(k), "label": f"Button {k}", "operations": ["CLICK"]}
                for k in range(1, n_elements + 1)]
    page = {"url": "https://shop.test/socks", "title": "Socks", "text": "sock " * 600}
    actions = [{"action": "Products", "kind": "click", "text": None, "page_changed": True}]
    return json.dumps({"page": page, "elements": elements, "recent_actions": actions})


def _row(id, site, task_type, question_type, options, gold, context, question, fmt="structured"):
    return dict(id=id, task_type=task_type, question_type=question_type, context=context, question=question,
                options=options, gold=gold, num_tokens=len(context) // 4, context_format=fmt, website=site)


ROWS = [
    _row("ap1", "a.com", AP, "operation", AP_OPTIONS, 1, _context(10), _question("Search for socks.")),
    _row("eg1", "b.com", EG, "click_target",
         ['3: {"element": "[3] Button 3", "current_value": ""}',
          '55: {"element": "[55] Button 55", "current_value": "", "role": "button"}'],
         1, _context(60), _question("Open the sale page.", "CLICK")),
    _row("eg2", "c.com", EG, "click_target",
         ["clickable: Socks | ancestor context: Shop", "clickable: Sale", "input: Search", "clickable: Cart"],
         1, "Website: c.com\n\nCurrent subgoal: open sale\n\nPage text:\nSale Socks Cart", "Which page element?",
         "text"),
    _row("ap2", "c.com", AP, "operation", AP_OPTIONS, 0, _context(5), _question("Open cart.")),
    _row("eg3", "a.com", EG, "select_target",
         ['11:1: {"element": "[11:1] Size → S", "current_value": "M", "role": "combobox"}',
          '11:2: {"element": "[11:2] Size → L", "current_value": "M", "role": "combobox"}'],
         0, _context(12), _question("Pick size S.", "SELECT")),
    _row("ap3", "d.com", AP, "operation", AP_OPTIONS, 3, _context(3), _question("Done.")),
]


@pytest.fixture
def instances(tmp_path):
    pq.write_table(pa.Table.from_pylist(ROWS), tmp_path / "webjev_mini.parquet")
    return {i.id: i for i in load_instances(tmp_path)}


def _synthetic(n_sites, per_site, types=(AP, EG)):
    return [Instance(f"{t}-{s}-{k}", f"site{s}.com", t, "q", ["A: a"], 0, "", "", 0, "text")
            for t in types for s in range(n_sites) for k in range(per_site)]


def test_load_instances_filters_task_types(tmp_path):
    pq.write_table(pa.Table.from_pylist(ROWS), tmp_path / "webjev_mini.parquet")
    assert len(load_instances(tmp_path)) == 6
    assert {i.id for i in load_instances(tmp_path, (EG,))} == {"eg1", "eg2", "eg3"}


def test_option_ids_for_action_and_grounding():
    assert option_id_and_label("CLICK: Click an element.", AP) == ("CLICK", "Click an element.")
    assert option_id_and_label(AP_OPTIONS[2], AP) == ("E72", "Scroll down within the region: Edition / US")
    grounding = '12: {"element": "[12] Try For Free", "current_value": ""}'
    assert option_id_and_label(grounding, EG) == ("E12", "Try For Free")
    assert option_id_and_label(
        '7: {"element": "[7] Search", "current_value": "socks", "role": "combobox", "expanded": "false"}', EG
    ) == ("E7", "combobox | Search | value=socks | expanded=false")
    assert option_id_and_label('11:2: {"element": "[11:2] Size → L", "current_value": ""}', EG) == ("E11:2", "Size → L")
    assert option_id_and_label("clickable: Sale | ctx: Shop", EG) == ("clickable", "Sale | ctx: Shop")


def test_compile_dedupes_option_ids(instances):
    ids = [o.id for o in compile_instance(instances["eg2"]).options]
    assert ids == ["clickable", "clickable_2", "input", "clickable_3"]
    dup = Instance("x", "s", AP, "operation", ["A: 1", "A: 2", "A_2: 3"], 2, "", "", 0, "text")
    assert [o.id for o in compile_instance(dup).options] == ["A", "A_2", "A_2_2"]


def test_compile_meta_gold_id_matches_gold_index(instances):
    for inst in instances.values():
        for mode in ("compact", "full"):
            state = compile_instance(inst, mode)
            assert state.meta["gold_id"] == state.options[inst.gold].id
    assert compile_instance(instances["eg2"]).meta["gold_id"] == "clickable_2"


def test_compile_compact_is_shorter_than_full(instances):
    for inst in instances.values():
        compact, full = compile_instance(inst, "compact"), compile_instance(inst, "full")
        assert full.state_text == inst.context + "\n\n" + inst.question
        assert (compact.state_text == full.state_text) if inst.context_format == "text" else (
            len(compact.state_text) < len(full.state_text))
        assert compact.instructions == full.instructions and compact.options == full.options


def test_compact_keeps_goal_subgoal_and_referenced_elements(instances):
    state = compile_instance(instances["eg1"], max_page_chars=100, max_elements=5)
    text = state.state_text
    assert "Task: Buy socks." in text and "Current subgoal: Open the sale page." in text
    assert "Operation: CLICK" in text and "URL: https://shop.test/socks" in text and "click: Products" in text
    assert "sock " * 20 in text and "sock " * 21 not in text
    listed = [line for line in text.splitlines() if line.startswith("[")]
    assert len(listed) == 5 and listed[-1].startswith("[55] button | Button 55")
    assert state.instructions == "Which element should the agent click?"
    assert compile_instance(instances["ap1"]).instructions == "Which operation should the agent take next?"


def test_site_split_is_disjoint_and_seeded():
    insts = _synthetic(20, 2)
    cal, held = site_split(insts, holdout_frac=0.3, seed=0)
    assert not cal & held and cal | held == {i.site for i in insts} and len(held) == 6
    assert site_split(insts, holdout_frac=0.3, seed=0) == (cal, held)
    assert site_split(list(reversed(insts)), holdout_frac=0.3, seed=0) == (cal, held)
    assert site_split(insts, holdout_frac=0.3, seed=1) != (cal, held)


def test_make_plan_balances_task_types():
    insts = _synthetic(10, 10)
    by_id = {i.id: i for i in insts}
    cal, held = site_split(insts)
    plan = make_plan(insts, cal, held, n_cal=20, n_test_in=10, n_test_shift=12)
    for split, n in [("cal", 20), ("test_in", 10), ("test_shift", 12)]:
        assert Counter(by_id[i].task_type for i in plan[split]) == {AP: n // 2, EG: n // 2}
    assert not set(plan["cal"]) & set(plan["test_in"])
    assert all(by_id[i].site in cal for i in plan["cal"] + plan["test_in"])
    assert all(by_id[i].site in held for i in plan["test_shift"])
    assert plan == make_plan(insts, cal, held, 20, 10, 12)
    assert plan != make_plan(insts, cal, held, 20, 10, 12, seed=1)


def test_make_plan_fills_from_other_type_when_short():
    insts = _synthetic(10, 10, (AP,)) + _synthetic(1, 3, (EG,))
    plan = make_plan(insts, {i.site for i in insts}, set(), n_cal=10, n_test_in=0, n_test_shift=4)
    assert Counter(i.split("-")[0] for i in plan["cal"]) == {AP: 7, EG: 3}
    assert plan["test_shift"] == []
