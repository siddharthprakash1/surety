import json
import random

import pytest

from surety.decide import MockBackend
from surety.env import Action, Element, Page
from surety.gate import CriticalDetector, CriticalOverride, FixedThreshold
from surety.loop import run
from surety.loop.runner import _bare
from surety.supervise import MockSupervisor
from surety.supervise.client import Usage
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

    def update(self, probs, gold, weight=1.0):
        pass


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


def test_unwrapped_policy_stops_on_executor_payment():
    env = FakeEnv([_page(Element("e1", "button", "Place order"))])
    ep = run("buy", env, always("e1"), FixedThreshold(0.5), MockSupervisor("e1"))
    assert ep.stop_reason == "stop:payment" and env.actions == [] and len(ep.steps) == 1


@pytest.mark.parametrize("label,critical", [("Send", "critical:send"), ("Forgot password?", "critical:credentials")])
def test_supervisor_chosen_critical_click_acts_and_is_recorded(label, critical):
    env = FakeEnv([_page(BUTTON, Element("e2", "button", label))])
    ep = run("mail", env, MockBackend(uniform), FixedThreshold(0.5), MockSupervisor("e2"), max_steps=1)
    assert env.actions == [Action("click", "e2")] and ep.steps[0]["critical"] == critical


@pytest.mark.parametrize("executor", ["confident", "uniform"])
@pytest.mark.parametrize("el", [Element("e2", "password", "pw"), Element("e2", "input", "Card number")])
def test_credential_entry_stops_without_typing(executor, el):
    env = FakeEnv([_page(BUTTON, el)])
    sup = MockSupervisor("e2", slots={"e2": "hunter2"})
    ep = run("login", env, always("e2") if executor == "confident" else MockBackend(uniform), FixedThreshold(0.5), sup)
    assert ep.stop_reason == "stop:credentials" and env.actions == [] and len(ep.steps) == 1
    assert ep.steps[0]["critical"] == "stop:credentials" and sup.calls == 1


def test_credential_field_labelled_with_another_category_still_stops():
    el = Element("e2", "input", "Enter your password to delete your account")
    env = FakeEnv([_page(BUTTON, el)])
    ep = run("go", env, MockBackend(uniform), FixedThreshold(0.5), MockSupervisor("e2", slots={"e2": "hunter2"}))
    assert ep.stop_reason == "stop:credentials" and env.actions == []


@pytest.mark.parametrize("value,stop,critical", [("Buy selected", "stop:payment", "stop:payment"),
                                                  ("Delete selected", None, "critical:destructive")])
def test_select_value_is_classified(value, stop, critical):
    env = FakeEnv([_page(BUTTON, Element("e2", "select", "Bulk actions"))])
    ep = run("go", env, MockBackend(uniform), FixedThreshold(0.5), MockSupervisor("e2", slots={"e2": value}),
             max_steps=1)
    assert ep.steps[0]["critical"] == critical
    assert env.actions == ([] if stop else [Action("select", "e2", value)])
    assert ep.stop_reason == (stop or "max_steps")


def test_executor_stop_is_recorded_as_critical():
    env = FakeEnv([_page(Element("e1", "button", "Place order"))])
    ep = run("buy", env, always("e1"), FixedThreshold(0.5), MockSupervisor("e1"))
    assert ep.steps[0]["critical"] == "stop:payment" and ep.steps[0]["chosen"] is None


class Recorder:
    name = "rec"

    def __init__(self, inner):
        self.inner, self.updates = inner, []

    def decide(self, probs):
        return self.inner.decide(probs)

    def update(self, probs, gold, weight=1.0):
        self.updates.append((gold, weight))


def test_calibrate_runs_before_the_gate_and_raw_probs_are_traced():
    flat = lambda p: {k: 1 / len(p) for k in p}  # noqa: E731
    env = FakeEnv([_page(BUTTON, LINK)])
    ep = run("go", env, always("e1"), FixedThreshold(0.5), MockSupervisor("e2"), max_steps=1, calibrate=flat)
    assert ep.steps[0]["decision"] == "escalate" and env.actions == [Action("click", "e2")]
    assert max(ep.steps[0]["probs_raw"].values()) == 0.9 and max(ep.steps[0]["probs"].values()) < 0.5


def test_escalation_updates_with_supervisor_label_weight_one():
    gate = Recorder(FixedThreshold(0.5))
    run("go", FakeEnv([_page(BUTTON, LINK)]), MockBackend(uniform), gate, MockSupervisor("e2"), max_steps=1)
    assert gate.updates == [("e2", 1.0)]


def test_unparsed_supervisor_answer_updates_with_weight_zero():
    gate = Recorder(FixedThreshold(0.5))
    run("go", FakeEnv([_page(BUTTON, LINK)]), MockBackend(uniform), gate, MockSupervisor(None), max_steps=1)
    assert [w for _, w in gate.updates] == [0.0]


def test_unaudited_acted_step_updates_with_weight_zero_and_no_supervisor_call():
    gate = Recorder(FixedThreshold(0.5))
    sup = MockSupervisor("e2")
    run("go", FakeEnv([_page(BUTTON, LINK)]), always("e1"), gate, sup, max_steps=1)
    assert gate.updates == [("e1", 0.0)] and sup.calls == 0


def test_audit_asks_full_view_acts_on_label_and_weights_one_over_r():
    gate = Recorder(FixedThreshold(0.5))
    env, sup = FakeEnv([_page(BUTTON, LINK)]), MockSupervisor("e2")
    ep = run("go", env, always("e1"), gate, sup, max_steps=1, audit_rate=0.5, rng=random.Random(3))  # 0.238 < 0.5
    assert sup.candidates == [None] and ep.steps[0]["audit"] == "e2"
    assert gate.updates == [("e2", 2.0)] and env.actions == [Action("click", "e2")]


class Abstains(MockSupervisor):
    def choose(self, state, candidate_ids=None):
        self.calls += 1
        self.candidates.append(candidate_ids)
        return None, Usage(0, 0, 0.0, abstained=True)


def test_audit_answered_none_is_a_label_and_is_acted_on():
    gate = Recorder(FixedThreshold(0.5))
    env = FakeEnv([_page(BUTTON, LINK)])
    ep = run("go", env, always("e1"), gate, Abstains(None), max_steps=1, audit_rate=0.5, rng=random.Random(3))
    assert gate.updates == [("none_of_these", 2.0)] and env.actions == [Action("scroll")]
    assert ep.steps[0]["audit"] == "none_of_these"


def test_shadow_label_is_recorded_but_not_acted_on_learnt_or_billed():
    gate = Recorder(FixedThreshold(0.5))
    env, sup = FakeEnv([_page(BUTTON, LINK)]), MockSupervisor("e2")
    ep = run("go", env, always("e1"), gate, sup, max_steps=1, shadow_rate=0.5, shadow_rng=random.Random(3))
    assert ep.steps[0]["shadow"] == "e2" and env.actions == [Action("click", "e1")]
    assert gate.updates == [("e1", 0.0)] and (ep.supervisor_calls, ep.shadow_calls) == (0, 1)


def test_shadows_do_not_move_audits():
    def audits(shadow_rate):
        pages = [_page(Element(f"e{i}", "button", f"B{i}"), LINK) for i in range(1, 9)]
        ep = run("go", FakeEnv(pages), MockBackend(confident(lambda s: s.options[0].id)), FixedThreshold(0.5),
                 MockSupervisor("e2"), max_steps=8, audit_rate=0.5, rng=random.Random(5), shadow_rate=shadow_rate,
                 shadow_rng=random.Random(9))
        return [i for i, s in enumerate(ep.steps) if "audit" in s]

    assert audits(0.0) == audits(1.0) and audits(0.0)


def test_orderings_follow_the_collection_rule():
    seen = []

    class Counting(MockBackend):
        def choice(self, state, n_perm=1, seed=0):
            seen.append((len(state.options), n_perm))
            return super().choice(state, n_perm, seed)

    many = [Element(f"e{i}", "button", f"B{i}") for i in range(20)]
    for page in ([BUTTON], many):
        run("go", FakeEnv([_page(*page)]), Counting(uniform), FixedThreshold(0.5), MockSupervisor(None), max_steps=1,
            n_perm=5, n_perm_large=3)
    assert seen == [(2, 5), (21, 3)]


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


def test_traced_url_drops_userinfo():
    assert _bare("https://u:hunter2@x.test:8080/s?q=1#f") == "https://x.test:8080/s"


def test_supervisor_gets_exact_candidate_subset():
    sup = MockSupervisor("e2")
    run("go", FakeEnv([_page(BUTTON, LINK, Element("e3", "link", "About"))]), MockBackend(), SubsetPolicy(), sup,
        max_steps=1)
    assert sup.candidates == [["e1", "e2"]]


def test_supervisor_candidates_in_page_order():
    class Reversed(SubsetPolicy):
        def decide(self, probs):
            return GateDecision("escalate", None, ["e3", "e1"], "set")

    sup = MockSupervisor("e1")
    run("go", FakeEnv([_page(BUTTON, LINK, Element("e3", "link", "About"))]), MockBackend(), Reversed(), sup,
        max_steps=1)
    assert sup.candidates == [["e1", "e3"]]


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


class AnswersEvery(MockSupervisor):
    """Answers only on calls k, 2k, ...; None (unparsed) otherwise."""

    def __init__(self, answer, k):
        super().__init__(answer)
        self.k = k

    def choose(self, state, candidate_ids=None):
        label, usage = super().choose(state, candidate_ids)
        return (label if self.calls % self.k == 0 else None), usage


def test_audit_weights_correct_for_unanswered_audits():
    # every audit drawn (r = 1); half the replies unparsed: answered ones weigh 1/(r·q̂), q̂ = answered / asked
    gate = Recorder(FixedThreshold(0.5))
    stats = [0, 0]
    run("go", FakeEnv([_page(BUTTON, LINK)]), always("e1"), gate, AnswersEvery("e2", 2), max_steps=4, audit_rate=1.0,
        rng=random.Random(0), audit_stats=stats)
    assert stats == [4, 2]
    assert gate.updates == [("e1", 0.0), ("e2", 2.0), ("e1", 0.0), ("e2", 2.0)]  # q̂ = 1/2 after calls 2 and 4
    gate = Recorder(FixedThreshold(0.5))
    run("go", FakeEnv([_page(BUTTON, LINK)]), always("e1"), gate, AnswersEvery("e2", 2), max_steps=2, audit_rate=1.0,
        rng=random.Random(0))
    assert gate.updates == [("e1", 0.0), ("e2", 1.0)]  # no stats: 1/r, as before


def test_a_dead_browser_is_raised_not_recorded_as_a_failed_step():
    from surety.env import EnvClosedError

    def die(env):
        raise EnvClosedError("Target page, context or browser has been closed")

    with pytest.raises(EnvClosedError):
        run("go", FakeEnv([_page(BUTTON, LINK)], after_act=die), always("e1"), FixedThreshold(0.5),
            MockSupervisor("e2"), max_steps=2)

    def reject(env):
        raise RuntimeError("element is not visible")

    ep = run("go", FakeEnv([_page(BUTTON, LINK)], after_act=reject), always("e1"), FixedThreshold(0.5),
             MockSupervisor("e2"), max_steps=2)
    assert ep.stop_reason == "error" and ep.steps[0]["error"] == "RuntimeError"
