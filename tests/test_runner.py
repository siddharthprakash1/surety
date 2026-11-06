import json

from surety.decide import MockBackend
from surety.env import Action, Element, Page
from surety.gate import CriticalDetector, CriticalOverride, FixedThreshold
from surety.loop import run
from surety.supervise import MockSupervisor
from surety.trace import Trace


class FakeEnv:
    def __init__(self, pages, terminate_after=None):
        self.pages = pages
        self.actions = []
        self.terminate_after = terminate_after

    @property
    def terminated(self):
        return self.terminate_after is not None and len(self.actions) >= self.terminate_after

    def observe(self):
        return self.pages[min(len(self.actions), len(self.pages) - 1)]

    def act(self, action):
        self.actions.append(action)

    def close(self):
        pass


def _page(*els, text=""):
    return Page("http://x.test", "T", text, list(els))


BUTTON = Element("e1", "button", "Next")
LINK = Element("e2", "link", "Home")


def confident(pick):
    def fn(state):
        top = pick(state)
        rest = (1 - 0.9) / (len(state.options) - 1)
        return {o.id: 0.9 if o.id == top else rest for o in state.options}
    return fn


def uniform(state):
    return {o.id: 1 / len(state.options) for o in state.options}


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
    ep = run("go", env, MockBackend(confident(lambda s: "e1")), FixedThreshold(0.5), sup)
    assert ep.stop_reason == "stuck" and len(ep.steps) == 4
    assert len(env.actions) == 3 and sup.calls == 1 and ep.steps[2]["supervisor"]


def test_payment_stop_does_not_act():
    env = FakeEnv([_page(Element("e1", "button", "Place order"))])
    gate = CriticalOverride(FixedThreshold(0.5), CriticalDetector())
    ep = run("buy", env, MockBackend(confident(lambda s: "e1")), gate, MockSupervisor("e1"))
    assert ep.stop_reason == "stop:payment" and env.actions == [] and len(ep.steps) == 1


def test_typing_uses_slot_value():
    env = FakeEnv([_page(Element("e1", "input", "Search"))])
    sup = MockSupervisor(None, slots={"e1": "shoes"})
    run("find shoes", env, MockBackend(confident(lambda s: "e1")), FixedThreshold(0.5), sup, max_steps=1)
    assert env.actions == [Action("type", "e1", "shoes")]


def test_escalated_none_reasks_full_view_then_scrolls():
    env = FakeEnv([_page(BUTTON, LINK)])
    ep = run("go", env, MockBackend(uniform), FixedThreshold(0.5), MockSupervisor("none_of_these"), max_steps=1)
    assert env.actions == [Action("scroll")] and ep.supervisor_calls == 2


def test_env_termination_stops():
    env = FakeEnv([_page(BUTTON)], terminate_after=1)
    ep = run("go", env, MockBackend(confident(lambda s: "e1")), FixedThreshold(0.5), MockSupervisor(None))
    assert ep.stop_reason == "terminated" and len(env.actions) == 1
