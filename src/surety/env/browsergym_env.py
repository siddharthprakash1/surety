import importlib

import numpy as np

from surety.env.page import Action, Element, Page

ROLES = {"link", "button", "textbox", "checkbox", "combobox", "menuitem", "option", "radio", "tab", "searchbox",
         "listbox", "spinbutton", "switch", "menuitemcheckbox", "menuitemradio", "slider", "treeitem", "password"}
MIN_VISIBILITY = 0.5
# clickable nodes outside ROLES (spans/divs with handlers: 42 % of MiniWoB++ tasks exposed 0–1 options without them)
NOT_CLICKABLE = {"StaticText", "InlineTextBox", "RootWebArea", "WebArea", "dialog", "alertdialog", "LineBreak"}
NAME_CHARS = 80
FALLBACK_NAMES = ("aria-label", "alt", "title", "placeholder", "name", "id", "class")  # clickable nodes outside ROLES
ROLE_FALLBACK_NAMES = ("placeholder", "name", "id")  # unnamed ROLES nodes (Chrome usually names them itself)
# Chrome's roles for date/time inputs: typed as text, in the ISO form Playwright's fill() accepts
DATE_FORMATS = {"Date": "YYYY-MM-DD", "DateTime": "YYYY-MM-DDTHH:MM", "InputTime": "HH:MM"}
TYPE_FORMATS = {"date": "YYYY-MM-DD", "datetime-local": "YYYY-MM-DDTHH:MM", "time": "HH:MM", "month": "YYYY-MM",
                "week": "YYYY-Www"}
# a <label> folds into these (its control), not into a link or button it happens to contain
FORM_ROLES = {"checkbox", "radio", "textbox", "searchbox", "combobox", "spinbutton", "password", "switch", "slider",
              "listbox"}
# autocomplete tokens of secret fields: an unnamed Stripe-style card input is only recognisable by these
SECRET_AUTOCOMPLETE = ("cc-", "one-time-code", "current-password", "new-password")


def _v(node: dict, key: str) -> str:
    return str((node.get(key) or {}).get("value", ""))


def _props(node: dict) -> dict:
    return {p["name"]: (p.get("value") or {}).get("value") for p in node.get("properties", [])}


def _is_password(role: str, props: dict) -> bool:
    return "password" in role.lower() or str(props.get("type") or "").lower() == "password"


def _dom_attrs(dom: dict) -> dict[str, dict[str, str]]:
    """bid -> DOM attributes from the CDP DOMSnapshot. Chrome's AX tree reports a password input as an unnamed
    textbox with no type (checked on MiniWoB login-user), so the input type and fallback names come from here."""
    s = dom.get("strings") or []
    out = {}
    for doc in dom.get("documents") or []:
        for attrs in (doc.get("nodes") or {}).get("attributes") or []:
            pairs = {s[attrs[k]]: s[attrs[k + 1]] for k in range(0, len(attrs) - 1, 2)}
            if "bid" in pairs:
                out[pairs["bid"]] = pairs
    return out


def _own_text(node: dict, by_id: dict, kept_bids: set, max_nodes: int = 400) -> tuple[str, list[str]]:
    """Text under `node` (first NAME_CHARS), not descending into other options (their text names them); and the
    options it contains, found past the text cap. Children sharing the node's own bid (a date input's spinbuttons)
    are part of it."""
    own_bid = node.get("browsergym_id")
    out, nested, stack, seen = [], [], list(reversed(node.get("childIds") or [])), 0
    while stack and seen < max_nodes:
        child, seen = by_id.get(stack.pop()), seen + 1
        if child is None:
            continue
        bid = child.get("browsergym_id")
        if bid in kept_bids and bid != own_bid:
            nested.append(bid)
            continue
        if _v(child, "role") == "StaticText" and sum(map(len, out)) < NAME_CHARS:
            out.append(_v(child, "name"))
        stack.extend(reversed(child.get("childIds") or []))
    return " ".join(" ".join(out).split())[:NAME_CHARS], nested


def _secret(attrs: dict) -> bool:
    return attrs.get("type", "").lower() == "password" or any(
        t.startswith(SECRET_AUTOCOMPLETE) for t in attrs.get("autocomplete", "").lower().split())


class BrowserGymEnv:
    def __init__(self, task_name: str, headless: bool = True):
        import gymnasium

        # gymnasium ids are registered by importing the benchmark package, e.g. browsergym.miniwob
        importlib.import_module("browsergym." + task_name.split(".")[0])
        self.env = gymnasium.make("browsergym/" + task_name, headless=headless)
        self.obs: dict = {}
        self.reward = 0.0
        self.terminated = self.truncated = False
        self._roles: dict[str, str] = {}

    def reset(self, seed: int | None = None) -> str:
        self.obs, _ = self.env.reset(seed=seed)
        self.reward, self.terminated, self.truncated = 0.0, False, False
        return self.obs.get("goal") or self.obs["goal_object"][0]["text"]

    def observe(self) -> Page:
        nodes = self.obs["axtree_object"]["nodes"]
        by_id = {n.get("nodeId"): n for n in nodes}
        extra = self.obs.get("extra_element_properties") or {}
        dom = _dom_attrs(self.obs.get("dom_object") or {})
        kept: dict[int, str] = {}  # id(node) -> role, in AX order; one node per bid (the first usable one)
        role_of: dict[str, str] = {}  # bid -> kept role
        for node in nodes:
            role, bid = _v(node, "role"), node.get("browsergym_id")
            if node.get("ignored") or bid is None or bid in role_of:
                continue
            props, attrs = _props(node), dom.get(str(bid), {})
            if props.get("disabled") or extra.get(bid, {}).get("visibility", 1.0) < MIN_VISIBILITY:
                continue
            if _is_password(role, props) or _secret(attrs):
                kept[id(node)] = "password"
            elif role in DATE_FORMATS or attrs.get("type", "").lower() in TYPE_FORMATS:
                kept[id(node)] = "textbox"
            elif role in ROLES:
                kept[id(node)] = "textbox" if role == "combobox" and props.get("editable") else role
            elif extra.get(bid, {}).get("clickable") and role not in NOT_CLICKABLE:
                kept[id(node)] = role if role not in ("generic", "none", "") else "clickable"
            if id(node) in kept:
                role_of[bid] = kept[id(node)]
        # labels first, so a control named by a <label> later in the page gets the name too
        by_dom_id = {a["id"]: b for b, a in dom.items() if a.get("id") and b in role_of}
        label_of: dict[str, str] = {}  # control bid -> text of the <label> that names it
        folded: set[int] = set()
        for node in nodes:
            if kept.get(id(node)) != "LabelText":
                continue
            own, nested = _own_text(node, by_id, set(role_of))
            attrs = dom.get(str(node.get("browsergym_id")), {})
            target = next((b for b in nested if role_of[b] in FORM_ROLES), None) or by_dom_id.get(attrs.get("for", ""))
            if target and role_of.get(target) in FORM_ROLES:
                folded.add(id(node))  # not an option of its own: its control is, named by it
                if text := _v(node, "name") or own:
                    label_of.setdefault(target, text)
        elements, text = [], []
        for node in nodes:
            role, name, bid = _v(node, "role"), _v(node, "name"), node.get("browsergym_id")
            if role == "StaticText" and name:
                text.append(name)
            if id(node) not in kept or id(node) in folded:
                continue
            ax_role, role, attrs, props = role, kept[id(node)], dom.get(str(bid), {}), _props(node)
            extra_role = ax_role not in ROLES and ax_role not in DATE_FORMATS and role != "password"  # clickable only
            if extra_role:
                own, nested = _own_text(node, by_id, set(role_of))
                name = name or own
                if not name and nested:
                    continue  # an unnamed wrapper around other options
            fallback = FALLBACK_NAMES if extra_role else ROLE_FALLBACK_NAMES
            name = (name or label_of.get(bid) or next((attrs[k] for k in fallback if attrs.get(k)), ""))[:NAME_CHARS]
            if not name and extra_role:
                continue
            if fmt := DATE_FORMATS.get(ax_role) or TYPE_FORMATS.get(attrs.get("type", "").lower()):
                name = f"{name} ({fmt})".strip()
            value = "" if role == "password" else str(props["checked"]) if "checked" in props else _v(node, "value")
            elements.append(Element(f"bid:{bid}", role, name, value))
        self._roles = {e.id: e.role for e in elements}
        titles = self.obs.get("open_pages_titles") or [""]
        title = titles[int(np.ravel(self.obs.get("active_page_index", 0))[0])]
        return Page(self.obs.get("url", ""), title, " ".join(text)[:4000], elements)

    def _step(self, action: str) -> None:
        self.obs, self.reward, self.terminated, self.truncated, _ = self.env.step(action)
        if self.obs.get("last_action_error"):
            raise RuntimeError(self.obs["last_action_error"])

    def act(self, action: Action) -> None:
        bid = (action.element_id or "").removeprefix("bid:")
        match action.kind:
            case "click":
                self._step(f"click({bid!r})")
            case "type":
                self._step(f"fill({bid!r}, {action.text or ''!r})")
                if self._roles.get(action.element_id) == "searchbox" and not self.terminated:
                    self._step(f"press({bid!r}, 'Enter')")
            case "select":
                self._step(f"select_option({bid!r}, {action.text or ''!r})")
            case "scroll":
                self._step("scroll(0, 600)")
            case "back":
                self._step("go_back()")
            case "done":
                self._step("send_msg_to_user('done')")

    def close(self) -> None:
        self.env.close()
