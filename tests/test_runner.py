import json

import pytest

from surety.decide import MockBackend
from surety.env import Action, Element, Page
from surety.gate import CriticalDetector, CriticalOverride, FixedThreshold
from surety.loop import run
from surety.supervise import MockSupervisor
from surety.trace import Trace
from surety.types import GateDecision


class FakeEnv:
    def __init__(self, pages, after_act=None):
        self.pages = pages
        self.actions = []
        self.after_act = after_act
        self.terminated = self.truncated = False

    def observe(self):
        return self.pages[min(len(self.actions), len(self.pages) - 1)]

    def act(self, action):
        self.actions.append(action)
        if self.after_act:
            self.after_act(self)

    def close(self):
        pass


class SubsetPolicy:
    name = "subset"

    def decide(self, probs):
        return GateDecision("escalate", None, ["e1", "e2"], "set")


def _page(*els, text="", url="http://x.test"):
    return Page(url, "T", text, list(els))


BUTTON = Element("e1", "button", "Next")
LINK = Element("e2", "link", "Home")
SEARCH = Element("e1", "input", "Search")


def confident(pick):
    def fn(state):
        top = pick(state)
        rest = (1 - 0.9) / (len(state.options) - 1)
        return {o.id: 0.9 if o.id == top else rest for o in state.options}
    return fn


def uniform(state):
    return {o.id: 1 / len(state.options) for o in state.options}


def always(option_id):
    return MockBackend(confident(lambda s: option_id))


def test_confident_reaches_done_without_supervisor(tmp_path):
    env = FakeEnv([_page(BUTTON), _page(BUTTON, text="Thanks")])
    backend = MockBackend(confident(lambda s: "none_of_these" if "Thanks" in s.state_text else "e1"))
    sup = MockSupervisor("e1")
    trace = Trace(tmp_path / "t.jsonl")
    ep = run("go", env, backend, FixedThreshold(0.5), sup, trace)
    trace.close()
    assert ep.stop_reason == "done" and len(ep.steps) <= 3
    assert ep.supervisor_calls == 0 == sup.calls and ep.executor_calls == 3
    assert env.actions == [Action("click", "e1"), Action("scroll"), Action("done")]
    rows = [json.loads(x) for x in (tmp_path / "t.jsonl").read_text().splitlines()]
    assert [r["kind"] for r in rows] == ["click", "scroll", "done"]


def test_low_confidence_acts_on_supervisor_choice():
    env = FakeEnv([_page(BUTTON, LINK)])
    ep = run("go", env, MockBackend(uniform), FixedThreshold(0.5), MockSupervisor("e2"), max_steps=1)
    assert env.actions == [Action("click", "e2")] and ep.supervisor_calls == 1
    assert ep.stop_reason == "max_steps" and ep.steps[0]["chosen"] == "e2"


def test_repeated_action_escalates_then_stuck():
    env = FakeEnv([_page(BUTTON)])
    sup = MockSupervisor("e1")
    ep = run("go", env, always("e1"), FixedThreshold(0.5), sup)
    assert ep.stop_reason == "stuck" and len(ep.steps) == 4
    assert len(env.actions) == 3 and sup.calls == 1 and ep.steps[2]["supervisor"]


def test_different_answer_on_third_repeat_resets_stuck():
    env = FakeEnv([_page(BUTTON, LINK)])
    ep = run("go", env, always("e1"), FixedThreshold(0.5), MockSupervisor("e2"), max_steps=6)
    assert [a.element_id for a in env.actions] == ["e1", "e1", "e2", "e1", "e1", "e2"]
    assert ep.stop_reason == "max_steps" and ep.supervisor_calls == 2


def test_payment_stop_does_not_act():
    env = FakeEnv([_page(Element("e1", "button", "Place order"))])
    gate = CriticalOverride(FixedThreshold(0.5), CriticalDetector())
    ep = run("buy", env, always("e1"), gate, MockSupervisor("e1"))
    assert ep.stop_reason == "stop:payment" and env.actions == [] and len(ep.steps) == 1


def test_supervisor_chosen_payment_stops_without_acting():
    env = FakeEnv([_page(SEARCH, Element("e2", "button", "Place order"))])
    ep = run("buy", env, MockBackend(uniform), FixedThreshold(0.5), MockSupervisor("e2"))
    assert ep.stop_reason == "stop:payment" and env.actions == [] and ep.supervisor_calls == 1


def test_supervisor_chosen_critical_acts_and_is_recorded():
    env = FakeEnv([_page(BUTTON, Element("e2", "button", "Send"))])
    ep = run("mail", env, MockBackend(uniform), FixedThreshold(0.5), MockSupervisor("e2"), max_steps=1)
    assert env.actions == [Action("click", "e2")] and ep.steps[0]["critical"] == "critical:send"


def test_typing_uses_slot_value():
    env = FakeEnv([_page(SEARCH)])
    sup = MockSupervisor(None, slots={"e1": "shoes"})
    ep = run("find shoes", env, always("e1"), FixedThreshold(0.5), sup, max_steps=1)
    assert env.actions == [Action("type", "e1", "shoes")] and ep.supervisor_calls == 1
    assert ep.steps[0]["supervisor"][0]["call"] == "slot_value"


def test_missing_slot_value_counts_as_none_of_these():
    env = FakeEnv([_page(SEARCH)])
    ep = run("find shoes", env, always("e1"), FixedThreshold(0.5), MockSupervisor(None))
    assert env.actions == [Action("scroll"), Action("done")] and ep.stop_reason == "done"


def test_typed_text_never_in_trace(tmp_path):
    env = FakeEnv([_page(SEARCH), _page(SEARCH, url="http://x.test/s?q=hunter2#hunter2")])
    trace = Trace(tmp_path / "t.jsonl")
    run("find", env, always("e1"), FixedThreshold(0.5), MockSupervisor(None, slots={"e1": "hunter2"}), trace,
        max_steps=2)
    trace.close()
    text = (tmp_path / "t.jsonl").read_text()
    assert "hunter2" not in text and '"url": "http://x.test/s"' in text


def test_supervisor_gets_exact_candidate_subset():
    sup = MockSupervisor("e2")
    run("go", FakeEnv([_page(BUTTON, LINK, Element("e3", "link", "About"))]), MockBackend(), SubsetPolicy(), sup,
        max_steps=1)
    assert sup.candidates == [["e1", "e2"]]


def test_subset_none_reasks_full_view_then_scrolls():
    env = FakeEnv([_page(BUTTON, LINK, Element("e3", "link", "About"))])
    sup = MockSupervisor("none_of_these")
    run("go", env, MockBackend(), SubsetPolicy(), sup, max_steps=1)
    assert sup.candidates == [["e1", "e2"], None] and env.actions == [Action("scroll")]


def test_full_view_none_scrolls_without_reask():
    env = FakeEnv([_page(BUTTON, LINK)])
    sup = MockSupervisor("none_of_these")
    ep = run("go", env, MockBackend(uniform), FixedThreshold(0.5), sup, max_steps=1)
    assert env.actions == [Action("scroll")] and sup.candidates == [None] and ep.supervisor_calls == 1


@pytest.mark.parametrize("flag", ["terminated", "truncated"])
def test_env_end_stops(flag):
    env = FakeEnv([_page(BUTTON)], after_act=lambda e: setattr(e, flag, True))
    ep = run("go", env, always("e1"), FixedThreshold(0.5), MockSupervisor(None))
    assert ep.stop_reason == flag and len(env.actions) == 1


def test_action_error_stops_after_recording(tmp_path):
    def boom(env):
        raise RuntimeError("element detached")

    trace = Trace(tmp_path / "t.jsonl")
    ep = run("go", FakeEnv([_page(BUTTON)], after_act=boom), always("e1"), FixedThreshold(0.5),
             MockSupervisor(None), trace)
    trace.close()
    assert ep.stop_reason == "error" and len(ep.steps) == 1 and ep.steps[0]["error"] == "RuntimeError"
    assert json.loads((tmp_path / "t.jsonl").read_text())["error"] == "RuntimeError"
