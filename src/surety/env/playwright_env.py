from surety.env.page import Action, Element, Page

SELECTOR = ("a, button, input, select, textarea, [role=button], [role=link], [role=textbox], "
            "[role=checkbox], [role=menuitem]")
MAX_ELEMENTS = 150
TIMEOUT_MS = 5_000
_TEXT_INPUTS = {"", "text", "email", "url", "tel", "number", "password", "date", "time", "datetime-local", "month",
                "week"}
# visible, inside the viewport and enabled, so scrolling changes what is enumerated
_JS = """els => els.map(el => {
  const r = el.getBoundingClientRect(), s = getComputedStyle(el);
  return {
    visible: r.width > 0 && r.height > 0 && r.bottom > 0 && r.top < innerHeight && !el.disabled
      && s.visibility !== 'hidden' && s.display !== 'none',
    tag: el.tagName.toLowerCase(), role: el.getAttribute('role') || '', type: (el.getAttribute('type') || '').toLowerCase(),
    aria: el.getAttribute('aria-label') || '', text: el.innerText || '', placeholder: el.getAttribute('placeholder') || '',
    name: el.getAttribute('name') || '', value: el.value == null ? '' : String(el.value),
    checked: typeof el.checked === 'boolean' ? String(el.checked) : (el.getAttribute('aria-checked') || ''),
  };
})"""


def _role(d: dict) -> str:
    tag = d["tag"]
    # native form controls: the tag decides, since role=combobox is often a plain text input
    if tag == "input":
        if d["type"] == "search" or d["role"] == "searchbox":
            return "searchbox"
        return "input" if d["type"] in _TEXT_INPUTS else d["type"]
    if tag in {"textarea", "select"}:
        return tag
    return d["role"] or ("link" if tag == "a" else tag)


def _element(i: int, d: dict) -> Element:
    role = _role(d)
    if role in {"checkbox", "radio"}:
        value = d["checked"]
    elif d["type"] == "password" or d["tag"] not in {"input", "textarea", "select"}:
        value = ""
    else:
        value = d["value"]
    text = " ".join(d["text"].split())[:80]
    return Element(f"e{i}", role, d["aria"] or text or d["placeholder"] or d["name"], value)


class PlaywrightEnv:
    def __init__(self, url: str, headless: bool = True):
        from playwright.sync_api import TimeoutError, sync_playwright

        self._timeout = TimeoutError
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=headless)
        self.page = self._browser.new_page()
        self.page.goto(url)
        self._handles: dict = {}
        self._roles: dict[str, str] = {}

    def observe(self) -> Page:
        loc = self.page.locator(SELECTOR)
        elements, self._handles = [], {}
        for i, d in enumerate(loc.evaluate_all(_JS)):
            if not d["visible"]:
                continue
            if len(elements) == MAX_ELEMENTS:
                break
            e = _element(len(elements) + 1, d)
            elements.append(e)
            self._handles[e.id] = loc.nth(i)
        self._roles = {e.id: e.role for e in elements}
        return Page(self.page.url, self.page.title(), self.page.inner_text("body")[:4000], elements)

    def act(self, action: Action) -> None:
        h = self._handles[action.element_id] if action.element_id else None
        match action.kind:
            case "click":
                h.click(timeout=TIMEOUT_MS)
            case "type":
                h.fill(action.text or "", timeout=TIMEOUT_MS)
                if self._roles.get(action.element_id) == "searchbox":
                    h.press("Enter", timeout=TIMEOUT_MS)
            case "select":
                h.select_option(action.text, timeout=TIMEOUT_MS)
            case "scroll":
                self.page.mouse.wheel(0, 600)
            case "back":
                self.page.go_back()
        try:
            self.page.wait_for_load_state("domcontentloaded", timeout=10_000)
        except self._timeout:
            pass

    def close(self) -> None:
        self._browser.close()
        self._pw.stop()
