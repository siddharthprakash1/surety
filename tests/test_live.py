import json
import random

import pytest

from surety.decide import MockBackend
from surety.env import Element, Page
from surety.eval.live import h4_table, run_episodes
from surety.supervise import MockSupervisor


class OneButton:
    """Clicking e1 solves the task (reward 1, terminated)."""

    def __init__(self, task):
        self.task, self.reward, self.terminated, self.truncated = task, 0.0, False, False

    def reset(self, seed=None):
        return f"press go ({self.task})"

    def observe(self):
        return Page("http://x.test", "T", "", [Element("e1", "button", "Go"), Element("e2", "link", "Home")])

    def act(self, action):
        if action.kind == "click" and action.element_id == "e1":
            self.reward, self.terminated = 1.0, True

    def close(self):
        pass


def _confident(state):
    return {o.id: 0.9 if o.id == "e1" else 0.1 / (len(state.options) - 1) for o in state.options}


def _cal_rows(n=200, seed=0):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        gold = rng.choice("abc")
        top = gold if rng.random() < 0.8 else rng.choice("abc")
        w = {o: (9.0 if o == top else rng.random()) for o in "abc"}
        z = sum(w.values())
        rows.append({"split": "cal", "gold_id": gold, "probs": {o: v / z for o, v in w.items()}})
    return rows


def _read(out):
    return [json.loads(x) for x in out.read_text().splitlines()]


def test_episodes_resumable_and_h4_table(tmp_path):
    out = tmp_path / "h4.jsonl"
    args = (["t1", "t2"], ["supervisor", "executor"], [0], OneButton, MockBackend(_confident), MockSupervisor("e1"), out)
    assert run_episodes(*args) == 4 and run_episodes(*args) == 0
    steps = [json.loads(x) for x in (tmp_path / "h4_traces" / "supervisor-0-t1.jsonl").read_text().splitlines()]
    assert [(s["decision"], s["chosen"]) for s in steps] == [("escalate", "e1")]
    recs = _read(out)
    by = {r["system"]: r for r in recs}
    assert by["supervisor"]["supervisor_calls"] == 1 and by["executor"]["supervisor_calls"] == 0
    assert all(r["reward"] == 1.0 and r["stop"] == "terminated" for r in recs)
    table = h4_table(recs)
    assert "| executor | executor | 2 | 1.000 | 0.000 | 2 | 0.000 | 0.000 | 1.000 | - | - | 0 | True |" in table
    assert "| supervisor | supervisor | 2 | 1.000 | - | 2 | 1.000 | - | 0.000 | - | 0.000 | 0 | - |" in table


def test_shadow_labels_estimate_live_autonomous_error(tmp_path):
    out = tmp_path / "h4.jsonl"
    # the supervisor disagrees with every acted step: live auto_err = acted share (1.0) × 1.0
    run_episodes(["t1", "t2", "t3"], ["executor"], [0], OneButton, MockBackend(_confident), MockSupervisor("e2"), out,
                 shadow_rate=1.0)
    recs = _read(out)
    assert [(r["labeled"], r["label_wrong"], r["shadow_calls"], r["supervisor_calls"]) for r in recs] == [(1, 1, 1, 0)] * 3
    assert "| 1.000 (n=3) |" in h4_table(recs)


def test_resume_restores_alpha_and_random_streams(tmp_path):
    tasks = [f"t{i}" for i in range(6)]
    run = lambda out, ts: run_episodes(ts, ["arisk"], [0], OneButton, MockBackend(_confident), MockSupervisor("e2"),  # noqa: E731
                                       out, rows=_cal_rows(), eps=0.1, audit_rate=0.5, max_steps=3)
    run(tmp_path / "straight.jsonl", tasks)
    run(tmp_path / "resumed.jsonl", tasks[:2])
    run(tmp_path / "resumed.jsonl", tasks)
    key = lambda r: (r["task"], r["acted"], r["labeled"], r["alpha_raw"], r["audit_rng"][1][0])  # noqa: E731
    straight, resumed = map(_read, (tmp_path / "straight.jsonl", tmp_path / "resumed.jsonl"))
    assert [key(r) for r in resumed] == [key(r) for r in straight] and len({r["alpha_raw"] for r in straight}) > 1


def test_resume_refuses_other_settings(tmp_path):
    out = tmp_path / "h4.jsonl"
    run_episodes(["t1"], ["executor"], [0], OneButton, MockBackend(_confident), MockSupervisor("e1"), out)
    with pytest.raises(ValueError):
        run_episodes(["t1"], ["executor"], [0], OneButton, MockBackend(_confident), MockSupervisor("e1"), out,
                     max_steps=20)


def test_failed_episodes_are_recorded_retried_and_stop_a_dead_run(tmp_path):
    out, broken = tmp_path / "h4.jsonl", {"t2"}

    def make(task):
        if task in broken:
            raise ConnectionError("server down")
        return OneButton(task)

    args = (["t1", "t2", "t3"], ["executor"], [0], make, MockBackend(_confident), MockSupervisor("e1"), out)
    assert run_episodes(*args) == 3
    assert [r["stop"] for r in _read(out)] == ["terminated", "error:ConnectionError", "terminated"]
    broken.clear()
    assert run_episodes(*args) == 1 and _read(out)[-1]["task"] == "t2"
    assert "| executor | executor | 3 |" in h4_table(_read(out))
    broken.update({"t4", "t5", "t6"})
    with pytest.raises(RuntimeError):
        run_episodes(["t4", "t5", "t6", "t7"], ["executor"], [0], make, MockBackend(_confident), MockSupervisor("e1"),
                     tmp_path / "dead.jsonl")


def test_h4_compares_on_matched_pairs(tmp_path):
    rec = lambda system, task, reward, calls: {"system": system, "spec": system, "seed": 0, "task": task,  # noqa: E731
                                               "reward": reward, "stop": "terminated", "steps": 1, "acted": 0,
                                               "supervisor_calls": calls, "labeled": 0, "label_wrong": 0,
                                               "choose_calls": calls, "unanswered": 0, "critical": []}
    recs = [rec("supervisor", "A", 1.0, 2), rec("supervisor", "B", 0.0, 2), rec("arisk", "A", 1.0, 0)]
    row = next(x for x in h4_table(recs).splitlines() if x.startswith("| arisk"))
    assert "| 1.000 | 0.000 | 1 |" in row  # success 1.0, Δ 0.0 on the one matched pair, not +0.5


class Rejects(OneButton):
    """The page rejects the click on e1 (as BrowserGym does with last_action_error)."""

    def act(self, action):
        raise RuntimeError("element not interactable")


def test_a_rejected_action_is_an_ordinary_failed_episode(tmp_path):
    out = tmp_path / "h4.jsonl"
    make = lambda task: Rejects(task) if task == "bad" else OneButton(task)  # noqa: E731
    args = (["good", "bad"], ["executor"], [0], make, MockBackend(_confident), MockSupervisor("e1"), out)
    assert run_episodes(*args) == 2 and run_episodes(*args) == 0  # not retried
    assert [r["stop"] for r in _read(out)] == ["terminated", "error"]
    row = next(x for x in h4_table(_read(out)).splitlines() if x.startswith("| executor"))
    assert row.startswith("| executor | executor | 2 | 0.500 |")


def test_resume_refuses_another_setup(tmp_path):
    out = tmp_path / "h4.jsonl"
    run_episodes(["t1"], ["executor"], [0], OneButton, MockBackend(_confident), MockSupervisor("e1"), out,
                 setup={"supervisor": {"model": "a"}})
    with pytest.raises(ValueError):
        run_episodes(["t1"], ["executor"], [0], OneButton, MockBackend(_confident), MockSupervisor("e1"), out,
                     setup={"supervisor": {"model": "b"}})


def test_h4_two_point_margin_is_inclusive():
    base = {"spec": "x", "seed": 0, "stop": "terminated", "steps": 1, "acted": 0, "supervisor_calls": 0,
            "critical": []}
    recs = [{**base, "system": "supervisor", "task": f"t{i}", "reward": 1.0, "supervisor_calls": 10} for i in range(50)]
    recs += [{**base, "system": "executor", "task": f"t{i}", "reward": float(i > 0)} for i in range(50)]
    assert h4_table(recs).splitlines()[-1].endswith("| True |")  # 49/50 vs 50/50 is exactly 2 points
