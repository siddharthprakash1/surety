import importlib

import numpy as np

from surety.env.page import Action, Element, Page

ROLES = {"link", "button", "textbox", "checkbox", "combobox", "menuitem", "option", "radio", "tab",
         "searchbox", "listbox"}


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
        self.terminated = False

    def reset(self, seed: int | None = None) -> str:
        self.obs, _ = self.env.reset(seed=seed)
        self.reward, self.terminated = 0.0, False
        return self.obs.get("goal") or self.obs["goal_object"][0]["text"]

    def observe(self) -> Page:
        elements, text = [], []
        for node in self.obs["axtree_object"]["nodes"]:
            role, name, bid = _v(node, "role"), _v(node, "name"), node.get("browsergym_id")
            if role == "StaticText" and name:
                text.append(name)
            if node.get("ignored") or bid is None or role not in ROLES:
                continue
            value = _v(node, "value")
            if role in {"checkbox", "radio"}:
                value = str(_props(node).get("checked", ""))
            elements.append(Element(f"bid:{bid}", role, name, value))
        titles = self.obs.get("open_pages_titles") or [""]
        title = titles[int(np.ravel(self.obs.get("active_page_index", 0))[0])]
        return Page(self.obs.get("url", ""), title, " ".join(text)[:4000], elements)

    def _step(self, action: str) -> None:
        self.obs, self.reward, self.terminated, _, _ = self.env.step(action)

    def act(self, action: Action) -> None:
        bid = (action.element_id or "").removeprefix("bid:")
        match action.kind:
            case "click":
                self._step(f"click({bid!r})")
            case "type":
                self._step(f"fill({bid!r}, {action.text or ''!r})")
                if not self.terminated:
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
