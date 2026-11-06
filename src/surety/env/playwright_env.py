from surety.env.page import Action, Element, Page

SELECTOR = ("a, button, input, select, textarea, [role=button], [role=link], [role=textbox], "
            "[role=checkbox], [role=menuitem]")
MAX_ELEMENTS = 150
_TEXT_INPUTS = {"", "text", "search", "email", "url", "tel", "number", "password", "date", "time", "datetime-local",
                "month", "week"}
_JS = """els => els.map(el => {
  const r = el.getBoundingClientRect(), s = getComputedStyle(el);
  return {
    visible: r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && s.display !== 'none',
    tag: el.tagName.toLowerCase(), role: el.getAttribute('role') || '', type: (el.getAttribute('type') || '').toLowerCase(),
    aria: el.getAttribute('aria-label') || '', text: el.innerText || '', placeholder: el.getAttribute('placeholder') || '',
    name: el.getAttribute('name') || '', value: el.value == null ? '' : String(el.value),
    href: typeof el.href === 'string' ? el.href : '',
  };
})"""


def _role(d: dict) -> str:
    if d["role"]:
        return d["role"]
    # an <input type=checkbox|submit|...> is clicked, not typed into
    if d["tag"] == "input" and d["type"] not in _TEXT_INPUTS:
        return d["type"]
    return d["tag"]


def _element(i: int, d: dict) -> Element:
    text = " ".join(d["text"].split())[:80]
    value = "" if d["type"] == "password" or d["tag"] not in {"input", "textarea", "select"} else d["value"]
    return Element(f"e{i}", _role(d), d["aria"] or text or d["placeholder"] or d["name"], value,
                   d["href"] if d["tag"] == "a" else "")


class PlaywrightEnv:
    def __init__(self, url: str, headless: bool = True):
        from playwright.sync_api import TimeoutError, sync_playwright

        self._timeout = TimeoutError
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=headless)
        self.page = self._browser.new_page()
        self.page.goto(url)
        self._handles: dict = {}

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
        return Page(self.page.url, self.page.title(), self.page.inner_text("body")[:4000], elements)

    def act(self, action: Action) -> None:
        h = self._handles[action.element_id] if action.element_id else None
        match action.kind:
            case "click":
                h.click()
            case "type":
                h.fill(action.text or "")
                h.press("Enter")
            case "select":
                h.select_option(action.text)
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
