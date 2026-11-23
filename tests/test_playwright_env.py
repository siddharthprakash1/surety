import pytest

from surety.env import Element
from surety.env.playwright_env import SELECTOR, _element


def _dom(**kw):
    d = dict(tag="input", role="", type="", aria="", text="", placeholder="", name="", value="", checked="")
    return {**d, **kw}


def test_autocomplete_card_field_is_secret_and_redacted():
    d = _dom(placeholder="1234 1234 1234 1234", autocomplete="cc-number", value="4242 4242")
    assert _element(1, d) == Element("e1", "password", "1234 1234 1234 1234")
    assert _element(2, _dom(autocomplete="billing cc-number")).role == "password"


@pytest.mark.parametrize("tag,role,expect", [("div", "combobox", "button"), ("div", "listbox", "button"),
                                             ("select", "listbox", "select"), ("input", "combobox", "input")])
def test_only_native_selects_are_selects(tag, role, expect):
    assert _element(1, _dom(tag=tag, role=role, text="Size")).role == expect


def test_name_falls_back_to_title_alt_id():
    assert _element(1, _dom(tag="img", title="", alt="Delete", id="x9")).name == "Delete"
    assert _element(2, _dom(tag="div", role="button", id="star-3")).name == "star-3"


def test_selector_covers_aria_widgets_and_inline_handlers():
    for part in ("[role=searchbox]", "[role=combobox]", "[role=tab]", "[role=option]", "[onclick]", "summary"):
        assert part in SELECTOR
