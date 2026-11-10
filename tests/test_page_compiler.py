from surety.compile import NONE_OF_THESE, compile_page
from surety.env import Element, Page


def _page(n=3):
    els = [Element(f"e{i}", "button", f"Button {i}") for i in range(1, n + 1)]
    els.append(Element("e99", "input", "Search", value="shoes"))
    return Page("http://x.test/a", "Shop", "hello " * 1000, els)


def test_option_ids_unique_and_none_last():
    s = compile_page(_page(), "buy shoes", [])
    ids = [o.id for o in s.options]
    assert len(ids) == len(set(ids)) and ids[-1] == NONE_OF_THESE


def test_element_cap():
    s = compile_page(_page(60), "t", [], max_elements=40)
    assert len(s.options) == 41 and s.options[39].id == "e40"
    assert s.meta == {"url": "http://x.test/a", "n_elements": 61, "task": "t"}


def test_label_format():
    labels = {o.id: o.label for o in compile_page(_page(), "t", []).options}
    assert labels["e1"] == "button: Button 1" and labels["e99"] == "input: Search [value=shoes]"


def test_state_text_and_instructions():
    hist = [f"h{i}" for i in range(10)]
    s = compile_page(_page(), "buy shoes", hist, max_text=20)
    assert s.instructions == "Which element should the agent act on next to accomplish: buy shoes"
    for head in ("TASK", "URL", "TITLE", "PAGE TEXT", "HISTORY"):
        assert head in s.state_text
    assert "hello " * 3 + "he" in s.state_text and "hello " * 4 not in s.state_text
    assert s.state_text.split("HISTORY\n")[1].splitlines() == hist[-8:]
