"""Real browser, scripted models: SURETY_LIVE=1 MINIWOB_URL=http://127.0.0.1:8765/miniwob/ (see scripts/run_miniwob.py)."""
import os

import pytest

if os.environ.get("SURETY_LIVE") != "1" or not os.environ.get("MINIWOB_URL"):
    pytest.skip("set SURETY_LIVE=1 and MINIWOB_URL to run MiniWoB++", allow_module_level=True)
pytest.importorskip("browsergym.miniwob")

from surety.decide import MockBackend  # noqa: E402
from surety.env import BrowserGymEnv  # noqa: E402
from surety.gate import FixedThreshold  # noqa: E402
from surety.loop import run  # noqa: E402
from surety.supervise.client import Usage  # noqa: E402

pytestmark = pytest.mark.live


@pytest.fixture(scope="module", autouse=True)
def _stop_browsergym_playwright():
    # BrowserGym keeps one sync Playwright per process; a later sync_playwright() (PlaywrightEnv) would refuse to start
    yield
    import browsergym.core as core

    if core._PLAYWRIGHT:
        core._PLAYWRIGHT.stop()
        core._set_global_playwright(None)


class Scripted:
    model, json_mode = "scripted", False

    def __init__(self, picks, slots):
        self.picks, self.slots = list(picks), slots

    def choose(self, state, candidate_ids=None):
        want = self.picks.pop(0) if self.picks else "\0"
        return next((o.id for o in state.options if o.label.endswith(want)), None), Usage(0, 0, 0.0)

    def slot_value(self, state, element_id):
        label = next(o.label for o in state.options if o.id == element_id)
        return next((v for k, v in self.slots.items() if label.endswith(k)), None), Usage(0, 0, 0.0)


def _uniform(state):
    return {o.id: 1 / len(state.options) for o in state.options}


def test_login_user_types_username_and_stops_at_password():
    env = BrowserGymEnv("miniwob.login-user")
    try:
        ep = run(env.reset(seed=0), env, MockBackend(_uniform), FixedThreshold(0.5),
                 Scripted(["username", "password"], {"username": "u", "password": "p"}), max_steps=5)
        assert ep.stop_reason == "stop:credentials"
        assert [s["kind"] for s in ep.steps] == ["type", None]
    finally:
        env.close()


def test_click_test_confident_executor_wins_without_supervisor():
    def confident(state):
        top = next(o.id for o in state.options if "Click" in o.label)
        return {o.id: 0.9 if o.id == top else 0.1 / (len(state.options) - 1) for o in state.options}

    env = BrowserGymEnv("miniwob.click-test")
    try:
        ep = run(env.reset(seed=0), env, MockBackend(confident), FixedThreshold(0.5), Scripted([], {}), max_steps=3)
        assert ep.stop_reason == "terminated" and env.reward == 1.0 and ep.supervisor_calls == 0
    finally:
        env.close()
