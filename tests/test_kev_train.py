import json
from collections import Counter

from surety.eval.kev_train import build_records, write_jsonl
from surety.eval.webjev import compile_instance
from surety.types import Instance

AP, EG = "action_prediction", "element_grounding"
AP_OPTIONS = ["CLICK: Click an element.", "TYPE_TEXT: Enter text.", "DONE: Every requirement is visibly satisfied."]
EG_OPTIONS = ['3: {"element": "[3] Button 3", "current_value": ""}',
              '5: {"element": "[5] Button 5", "current_value": "", "role": "button"}']


def _context():
    elements = [{"role": "button", "value": "", "index": str(k), "label": f"Button {k}", "operations": ["CLICK"]}
                for k in range(1, 6)]
    page = {"url": "https://shop.test/socks", "title": "Socks", "text": "sock " * 50}
    actions = [{"action": "Products", "kind": "click", "text": None, "page_changed": True}]
    return json.dumps({"page": page, "elements": elements, "recent_actions": actions})


def _question():
    return json.dumps({"goal": "The user's original task (global constraints; no stage may drop any of its conditions): "
                               "Buy socks.", "operation": "CLICK"})


def _inst(id, site, task_type, options=None):
    options = options or (AP_OPTIONS if task_type == AP else EG_OPTIONS)
    return Instance(id=id, site=site, task_type=task_type, question_type="operation" if task_type == AP else "click_target",
                    options=options, gold=1, context=_context(), question=_question(), num_tokens=0,
                    context_format="structured")


def _pool(sites, per_site):
    return [_inst(f"{t[:2]}-{s}-{k}", s, t) for t in (AP, EG) for s in sites for k in range(per_site)]


def _plan(cal_sites, held_sites, cal=(), test_in=(), test_shift=()):
    return {"seed": 0, "cal_sites": list(cal_sites), "held_sites": list(held_sites), "cal": list(cal),
            "test_in": list(test_in), "test_shift": list(test_shift), "max_options": 64}


def _ids(records):
    return [r["_meta"]["id"] for r in records]


def test_build_records_excludes_plan_ids_and_held_sites():
    insts = _pool(["a.com", "b.com", "h.com"], 4) + [_inst("big", "a.com", AP, AP_OPTIONS * 30)]
    plan = _plan(["a.com", "b.com"], ["h.com"], cal=["ac-a.com-0", "el-b.com-1"], test_in=["ac-b.com-2"],
                 test_shift=["ac-h.com-0"])
    excluded = {"ac-a.com-0", "el-b.com-1", "ac-b.com-2", "big"}
    eligible = {i.id for i in insts if i.site != "h.com"} - excluded
    train, heldout = build_records(insts, plan, n_train=100, n_heldout=0, seed=0, max_options=64)
    assert set(_ids(train)) == eligible and heldout == []
    train, heldout = build_records(insts, plan, n_train=100, n_heldout=3, seed=0, max_options=64)
    assert len(heldout) == 3 and set(_ids(train)) | set(_ids(heldout)) <= eligible


def test_build_records_heldout_sites_disjoint_from_train_sites():
    sites = [f"s{k}.com" for k in range(6)]
    insts = _pool(sites, 4)
    site_of = {i.id: i.site for i in insts}
    train, heldout = build_records(insts, _plan(sites, []), n_train=100, n_heldout=10, seed=0, heldout_site_cap=3)
    train_sites, heldout_sites = {site_of[i] for i in _ids(train)}, {site_of[i] for i in _ids(heldout)}
    assert len(heldout) == 10 and len(train) == 16
    assert len(train_sites) == 2 and train_sites.isdisjoint(heldout_sites)
    assert max(Counter(site_of[i] for i in _ids(heldout)).values()) <= 3


def test_build_records_trains_only_on_a_spread_plans_train_sites():
    insts = _pool(["t1.com", "t2.com", "c.com", "h.com"], 4)
    site_of = {i.id: i.site for i in insts}
    plan = _plan(["c.com"], ["h.com"]) | {"train_sites": ["t1.com", "t2.com"]}
    train, heldout = build_records(insts, plan, n_train=100, n_heldout=0, seed=0)
    assert {site_of[i] for i in _ids(train)} == {"t1.com", "t2.com"} and len(train) == 16 and heldout == []


def test_build_records_holds_out_whole_websites_on_a_spread_plan():
    hosts = [f"{h}.w{s}.com" for s in range(6) for h in ("a", "b")]
    insts = _pool(hosts, 2)
    site_of = {i.id: i.site for i in insts}
    plan = _plan([], []) | {"train_sites": hosts}
    train, heldout = build_records(insts, plan, n_train=100, n_heldout=8, seed=0, heldout_site_cap=4)
    websites = lambda rs: {site_of[i].split(".", 1)[1] for i in _ids(rs)}
    assert len(heldout) == 8 and len(websites(heldout)) == 2 and websites(train).isdisjoint(websites(heldout))
    assert max(Counter(site_of[i].split(".", 1)[1] for i in _ids(heldout)).values()) <= 4


def test_build_records_shape_matches_kev_format(tmp_path):
    insts = _pool(["a.com", "b.com"], 3)
    train, heldout = build_records(insts, _plan(["a.com", "b.com"], []), n_train=4, n_heldout=2, seed=0)
    by_id = {i.id: i for i in insts}
    for r in train + heldout:
        assert set(r) == {"state", "questions", "_meta"}
        (qid, q), = r["questions"].items()
        inst = by_id[r["_meta"]["id"]]
        state = compile_instance(inst, "compact")
        assert qid == {AP: "action", EG: "element"}[inst.task_type]
        assert q == {"type": "choice", "instructions": state.instructions,
                     "criteria": {o.id: o.label for o in state.options}, "label": state.meta["gold_id"], "src": "webjev"}
        assert q["label"] in q["criteria"] and r["state"] == state.state_text
    write_jsonl(train, tmp_path / "out" / "train.jsonl")
    assert [json.loads(line) for line in (tmp_path / "out" / "train.jsonl").read_text().splitlines()] == train


def test_build_records_balances_task_types():
    insts = _pool([f"s{k}.com" for k in range(5)], 4)
    plan = _plan([f"s{k}.com" for k in range(5)], [])
    train, heldout = build_records(insts, plan, n_train=10, n_heldout=4, seed=0)
    assert Counter(next(iter(r["questions"])) for r in train) == {"action": 5, "element": 5}
    assert Counter(next(iter(r["questions"])) for r in heldout) == {"action": 2, "element": 2}
    assert (train, heldout) == build_records(list(reversed(insts)), plan, 10, 4, seed=0)
    assert _ids(train) != _ids(build_records(insts, plan, 10, 4, seed=1)[0])
    short = [i for i in insts if i.task_type == AP] + [i for i in insts if i.task_type == EG][:2]
    train, _ = build_records(short, plan, n_train=10, n_heldout=0, seed=0)
    assert Counter(next(iter(r["questions"])) for r in train) == {"action": 8, "element": 2}
