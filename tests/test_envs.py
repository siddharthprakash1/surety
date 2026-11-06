import numpy as np

from surety.env import Action, BrowserGymEnv, Element
from surety.env.playwright_env import _element


def _dom(**kw):
    d = dict(visible=True, tag="input", role="", type="", aria="", text="", placeholder="", name="", value="",
             href="")
    return {**d, **kw}


def test_playwright_element_redacts_password_and_maps_roles():
    assert _element(1, _dom(type="password", name="pw", value="hunter2")) == Element("e1", "input", "pw")
    assert _element(2, _dom(type="checkbox", aria="Agree")).role == "checkbox"
    a = _element(3, _dom(tag="a", text="  Home \n page ", href="http://x/h", value="v"))
    assert a == Element("e3", "a", "Home page", "", "http://x/h")


OBS = {
    "axtree_object": {"nodes": [
        {"role": {"value": "button"}, "name": {"value": "OK"}, "browsergym_id": "12"},
        {"role": {"value": "StaticText"}, "name": {"value": "Click OK"}},
        {"role": {"value": "checkbox"}, "name": {"value": "A"}, "browsergym_id": "13",
         "properties": [{"name": "checked", "value": {"value": "true"}}]},
        {"role": {"value": "generic"}, "browsergym_id": "14"},
        {"role": {"value": "link"}, "name": {"value": "x"}, "browsergym_id": "15", "ignored": True},
    ]},
    "url": "http://m/", "open_pages_titles": ["T"], "active_page_index": np.array([0]),
    "goal_object": [{"type": "text", "text": "Click OK"}],
}


class FakeGym:
    def __init__(self):
        self.steps = []

    def reset(self, seed=None):
        return OBS, {}

    def step(self, a):
        self.steps.append(a)
        return OBS, 1.0, False, False, {}


def _bg():
    env = object.__new__(BrowserGymEnv)
    env.env = FakeGym()
    return env


def test_browsergym_observe_converts_axtree():
    env = _bg()
    assert env.reset() == "Click OK"
    p = env.observe()
    assert (p.url, p.title, p.text) == ("http://m/", "T", "Click OK")
    assert p.elements == [Element("bid:12", "button", "OK"), Element("bid:13", "checkbox", "A", "true")]


def test_browsergym_action_strings():
    env = _bg()
    for a in [Action("click", "bid:12"), Action("type", "bid:12", "it's"), Action("select", "bid:3", "b"),
              Action("scroll"), Action("back"), Action("done")]:
        env.act(a)
    assert env.env.steps == ["click('12')", "fill('12', \"it's\")", "press('12', 'Enter')",
                             "select_option('3', 'b')", "scroll(0, 600)", "go_back()", "send_msg_to_user('done')"]
    assert env.reward == 1.0 and env.terminated is False
