import numpy as np
import pytest

from surety.env import Action, BrowserGymEnv, Element
from surety.env.playwright_env import _element


def _dom(**kw):
    d = dict(tag="input", role="", type="", aria="", text="", placeholder="", name="", value="", checked="")
    return {**d, **kw}


def test_playwright_password_role_and_redacted_value():
    assert _element(1, _dom(type="password", name="pw", value="hunter2")) == Element("e1", "password", "pw")


@pytest.mark.parametrize("dom,role", [
    (_dom(role="combobox", name="q"), "input"),
    (_dom(type="search"), "searchbox"),
    (_dom(type="submit"), "submit"),
    (_dom(tag="select", role="listbox"), "select"),
    (_dom(tag="a"), "link"),
    (_dom(tag="div", role="menuitem"), "menuitem"),
])
def test_playwright_role(dom, role):
    assert _element(1, dom).role == role


def test_playwright_element_name_and_checked():
    a = _element(3, _dom(tag="a", text="  Home \n page ", value="v"))
    assert a == Element("e3", "link", "Home page")
    assert _element(4, _dom(type="checkbox", aria="Agree", value="on", checked="true")).value == "true"


def _node(role, bid=None, name="", props=None, **kw):
    n = {"role": {"value": role}, "name": {"value": name}, **kw}
    if bid:
        n["browsergym_id"] = bid
    if props:
        n["properties"] = [{"name": k, "value": {"value": v}} for k, v in props.items()]
    return n


def _obs(error=""):
    return {
        "axtree_object": {"nodes": [
            _node("button", "12", "OK"),
            _node("StaticText", name="Click OK"),
            _node("checkbox", "13", "A", {"checked": "true"}),
            _node("generic", "14"),
            _node("link", "15", "x", ignored=True),
            _node("combobox", "16", "Search", {"editable": "plaintext"}),
            _node("combobox", "17", "Size"),
            _node("searchbox", "18", "Find"),
            _node("button", "19", "Off", {"disabled": True}),
            _node("button", "20", "Below"),
            _node("spinbutton", "21", "Qty"),
            _node("textbox", "22", "Secret", {"type": "password"}, value={"value": "hunter2"}),
        ]},
        "extra_element_properties": {"20": {"visibility": 0.0}, "12": {"visibility": 1.0}},
        "url": "http://m/", "open_pages_titles": ["T"], "active_page_index": np.array([0]),
        "goal_object": [{"type": "text", "text": "Click OK"}], "last_action_error": error,
    }


class FakeGym:
    def __init__(self, error=""):
        self.steps = []
        self.error = error

    def reset(self, seed=None):
        return _obs(), {}

    def step(self, a):
        self.steps.append(a)
        return _obs(self.error), 1.0, False, True, {}


def _bg(error=""):
    env = object.__new__(BrowserGymEnv)
    env.env = FakeGym(error)
    env.reset()
    return env


def test_browsergym_observe_converts_axtree():
    p = _bg().observe()
    assert (p.url, p.title, p.text) == ("http://m/", "T", "Click OK")
    assert p.elements == [
        Element("bid:12", "button", "OK"), Element("bid:13", "checkbox", "A", "true"),
        Element("bid:16", "textbox", "Search"), Element("bid:17", "combobox", "Size"),
        Element("bid:18", "searchbox", "Find"), Element("bid:21", "spinbutton", "Qty"),
        Element("bid:22", "password", "Secret"),
    ]


def test_browsergym_action_strings():
    env = _bg()
    env.observe()
    for a in [Action("click", "bid:12"), Action("type", "bid:16", "it's"), Action("type", "bid:18", "x"),
              Action("select", "bid:17", "b"), Action("scroll"), Action("back"), Action("done")]:
        env.act(a)
    assert env.env.steps == ["click('12')", "fill('16', \"it's\")", "fill('18', 'x')", "press('18', 'Enter')",
                             "select_option('17', 'b')", "scroll(0, 600)", "go_back()", "send_msg_to_user('done')"]
    assert env.reward == 1.0 and env.terminated is False and env.truncated is True


def test_browsergym_action_error_raises():
    env = _bg("no element")
    with pytest.raises(RuntimeError):
        env.act(Action("click", "bid:12"))
