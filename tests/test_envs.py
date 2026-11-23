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


def _snapshot(*inputs):
    strings, attributes = [], []
    for attrs in inputs:
        row = []
        for k, v in attrs.items():
            row += [len(strings), len(strings) + 1]
            strings += [k, v]
        attributes.append(row)
    return {"strings": strings, "documents": [{"nodes": {"attributes": attributes}}]}


def test_browsergym_password_type_and_names_come_from_dom_snapshot():
    # as Chrome reports MiniWoB login-user: unnamed textboxes, the input type only in the DOM
    env = _bg()
    env.obs = {**_obs(), "axtree_object": {"nodes": [_node("textbox", "16"), _node("textbox", "19"),
                                                     _node("textbox", "21", value={"value": "4242"})]},
               "dom_object": _snapshot({"type": "text", "id": "username", "bid": "16"},
                                       {"type": "password", "id": "password", "bid": "19"},
                                       {"placeholder": "1234 1234 1234 1234", "autocomplete": "cc-number", "bid": "21"})}
    assert env.observe().elements == [Element("bid:16", "textbox", "username"),
                                      Element("bid:19", "password", "password"),
                                      Element("bid:21", "password", "1234 1234 1234 1234")]


def test_browsergym_clickable_nodes_named_by_own_text_or_dom():
    # MiniWoB click-link / email-inbox: span links and clickable rows are generic nodes with text children
    env = _bg()
    nodes = [_node("generic", "30", nodeId="a", childIds=["b"]), _node("StaticText", name="et", nodeId="b"),
             _node("generic", "31", nodeId="c", childIds=["d", "e"]), _node("StaticText", name="Alice: hi", nodeId="d"),
             _node("image", "32", nodeId="e"),
             _node("generic", "33", nodeId="f", childIds=["g"]), _node("button", "34", "OK", nodeId="g"),
             _node("generic", "35", nodeId="h"), _node("generic", "36", nodeId="i", childIds=["j"]),
             _node("StaticText", name="not clickable", nodeId="j")]
    clickable = {b: {"clickable": True} for b in ("30", "31", "32", "33", "35")}
    env.obs = {**_obs(), "axtree_object": {"nodes": nodes}, "extra_element_properties": clickable,
               "dom_object": _snapshot({"class": "trash", "bid": "32"})}
    assert env.observe().elements == [Element("bid:30", "clickable", "et"), Element("bid:31", "clickable", "Alice: hi"),
                                      Element("bid:32", "image", "trash"), Element("bid:34", "button", "OK")]


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


def test_browsergym_labels_name_their_controls_instead_of_duplicating_them():
    # MiniWoB click-checkboxes: <label><input type=checkbox>Nb</label>; and a <label for=x> pointing at a textbox
    env = _bg()
    nodes = [_node("LabelText", "40", nodeId="a", childIds=["b", "c"]), _node("checkbox", "41", nodeId="b"),
             _node("StaticText", name="Nb", nodeId="c"),
             _node("LabelText", "42", nodeId="d", childIds=["e"]), _node("StaticText", name="City", nodeId="e"),
             _node("textbox", "43", nodeId="f")]
    env.obs = {**_obs(), "axtree_object": {"nodes": nodes},
               "extra_element_properties": {"40": {"clickable": True}, "42": {"clickable": True}},
               "dom_object": _snapshot({"for": "city", "bid": "42"}, {"id": "city", "bid": "43"})}
    assert env.observe().elements == [Element("bid:41", "checkbox", "Nb"), Element("bid:43", "textbox", "City")]


def test_browsergym_one_option_per_bid_and_tokenised_autocomplete():
    env = _bg()
    nodes = [_node("textbox", "50", "Date"), _node("spinbutton", "50", "Day"), _node("textbox", "51")]
    env.obs = {**_obs(), "axtree_object": {"nodes": nodes},
               "dom_object": _snapshot({"autocomplete": "billing cc-number", "bid": "51"})}
    assert [(e.id, e.role) for e in env.observe().elements] == [("bid:50", "textbox"), ("bid:51", "password")]


def test_browsergym_role_nodes_keep_their_fallback_names():
    env = _bg()
    env.obs = {**_obs(), "axtree_object": {"nodes": [_node("textbox", "60")]},
               "dom_object": _snapshot({"title": "Search the site", "class": "form-control", "bid": "60"})}
    assert env.observe().elements == [Element("bid:60", "textbox", "")]


def test_browsergym_date_and_time_inputs_are_typed_with_their_format():
    # Chrome: a Date node whose spinbutton children share its bid; a time input seen as a textbox
    env = _bg()
    nodes = [_node("Date", "70", nodeId="a", childIds=["b", "c"]), _node("spinbutton", "70", "mm", nodeId="b"),
             _node("spinbutton", "70", "dd", nodeId="c"), _node("textbox", "71", "Start")]
    env.obs = {**_obs(), "axtree_object": {"nodes": nodes}, "extra_element_properties": {"70": {"clickable": True}},
               "dom_object": _snapshot({"type": "time", "bid": "71"})}
    assert env.observe().elements == [Element("bid:70", "textbox", "(YYYY-MM-DD)"),
                                      Element("bid:71", "textbox", "Start (HH:MM)")]


def test_browsergym_label_folding_edge_cases():
    env = _bg()
    long_text = "x" * 100
    nodes = [_node("LabelText", "80", nodeId="a", childIds=["b", "c"]), _node("StaticText", name=long_text, nodeId="b"),
             _node("checkbox", "81", nodeId="c"),
             _node("LabelText", "82", nodeId="d", childIds=["e", "f"]), _node("StaticText", name="Pro plan", nodeId="e"),
             _node("link", "83", "details", nodeId="f"),
             _node("clickable", "84", nodeId="g"),  # a control before its <label for>
             _node("LabelText", "85", nodeId="h"),  # empty label for the same control
             _node("LabelText", "86", nodeId="i", childIds=["j"]), _node("StaticText", name="I agree", nodeId="j")]
    env.obs = {**_obs(), "axtree_object": {"nodes": nodes},
               "extra_element_properties": {b: {"clickable": True} for b in ("80", "82", "84", "85", "86")},
               "dom_object": _snapshot({"type": "checkbox", "bid": "81"}, {"id": "agree", "bid": "84"},
                                       {"for": "agree", "bid": "85"}, {"for": "agree", "bid": "86"})}
    by_id = {e.id: e for e in env.observe().elements}
    assert by_id["bid:81"].name == "x" * 80 and "bid:80" not in by_id          # long label folded
    assert by_id["bid:82"].name == "Pro plan" and by_id["bid:83"].name == "details"  # label around a link kept


def test_browsergym_month_and_week_inputs_get_their_own_format():
    # Chrome gives month, week and datetime-local inputs the same DateTime role: the input type decides the hint
    env = _bg()
    nodes = [_node("DateTime", "95", "Month"), _node("DateTime", "96", "Week"), _node("DateTime", "97", "When")]
    env.obs = {**_obs(), "axtree_object": {"nodes": nodes},
               "dom_object": _snapshot({"type": "month", "bid": "95"}, {"type": "week", "bid": "96"},
                                       {"type": "datetime-local", "bid": "97"})}
    assert [e.name for e in env.observe().elements] == ["Month (YYYY-MM)", "Week (YYYY-Www)",
                                                         "When (YYYY-MM-DDTHH:MM)"]


def test_browsergym_label_for_wins_over_a_nested_control():
    # <label for=email>Email <input type=checkbox></label> <input id=email>: HTML names the email box, not the checkbox
    env = _bg()
    nodes = [_node("LabelText", "90", nodeId="a", childIds=["b", "c"]), _node("StaticText", name="Email", nodeId="b"),
             _node("checkbox", "91", nodeId="c"), _node("textbox", "92", nodeId="d")]
    env.obs = {**_obs(), "axtree_object": {"nodes": nodes}, "extra_element_properties": {"90": {"clickable": True}},
               "dom_object": _snapshot({"for": "email", "bid": "90"}, {"type": "checkbox", "bid": "91"},
                                       {"id": "email", "bid": "92"})}
    assert env.observe().elements == [Element("bid:91", "checkbox", ""), Element("bid:92", "textbox", "Email")]


def test_browsergym_dead_browser_raises_env_closed():
    from surety.env import EnvClosedError

    env = _bg()
    env.env.step = lambda action: ({**_obs(), "last_action_error": "TargetClosedError: Target page, context or "
                                    "browser has been closed"}, 0.0, False, False, {})
    with pytest.raises(EnvClosedError):
        env.act(Action("click", "bid:12"))
    env.env.step = lambda action: ({**_obs(), "last_action_error": "Error: element is not visible"}, 0.0, False,
                                   False, {})
    with pytest.raises(RuntimeError) as e:
        env.act(Action("click", "bid:12"))
    assert not isinstance(e.value, EnvClosedError)
