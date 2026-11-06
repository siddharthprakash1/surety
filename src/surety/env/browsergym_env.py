import importlib

import numpy as np

from surety.env.page import Action, Element, Page

ROLES = {"link", "button", "textbox", "checkbox", "combobox", "menuitem", "option", "radio", "tab", "searchbox",
         "listbox", "spinbutton", "switch", "menuitemcheckbox", "menuitemradio", "slider", "treeitem"}
MIN_VISIBILITY = 0.5


def _v(node: dict, key: str) -> str:
    return str((node.get(key) or {}).get("value", ""))


def _props(node: dict) -> dict:
    return {p["name"]: (p.get("value") or {}).get("value") for p in node.get("properties", [])}


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
        extra = self.obs.get("extra_element_properties") or {}
        elements, text = [], []
        for node in self.obs["axtree_object"]["nodes"]:
            role, name, bid = _v(node, "role"), _v(node, "name"), node.get("browsergym_id")
            if role == "StaticText" and name:
                text.append(name)
            if node.get("ignored") or bid is None or role not in ROLES:
                continue
            props = _props(node)
            if props.get("disabled") or extra.get(bid, {}).get("visibility", 1.0) < MIN_VISIBILITY:
                continue
            if role == "combobox" and props.get("editable"):
                role = "textbox"
            value = str(props["checked"]) if "checked" in props else _v(node, "value")
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
