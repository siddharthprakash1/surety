from types import SimpleNamespace

import pytest

from surety.gate import AdaptiveConformal, CriticalDetector, CriticalOverride, FixedThreshold, SplitConformal
from surety.types import GateDecision


@pytest.mark.parametrize(
    "label,url,category",
    [
        ("Place Order", "", "payment"),
        ("BUY NOW", "", "payment"),
        ("Buy", "", "payment"),
        ("Proceed to payment", "", "payment"),
        ("PayPal", "", "payment"),
        ("Check out", "", "payment"),
        ("Send payment", "", "payment"),
        ("Continue", "https://shop.test/checkout", "payment"),
        ("Delete repository", "", "destructive"),
        ("Removed", "", "destructive"),
        ("Unsubscribe", "", "destructive"),
        ("Send", "", "send"),
        ("Log out", "", "account"),
        ("Logout", "", "account"),
        ("Sign-out", "", "account"),
        ("Change password", "", "account"),
        ("Password", "", "credentials"),
        ("Passcode", "", "credentials"),
        ("Card number", "", "credentials"),
        ("CVV", "", "credentials"),
        ("cvc code", "", "credentials"),
        ("Social Security number", "", "credentials"),
        ("SSN", "", "credentials"),
        ("Enter one-time code", "", "credentials"),
        ("OTP", "", "credentials"),
        ("Expiry date", "", "credentials"),
        ("Submit order", "", "payment"),
        ("Complete your order", "", "payment"),
        ("Confirm payment", "", "payment"),
        ("Expiration date", "", "credentials"),
        ("Security code", "", "credentials"),
        ("CVV2", "", "credentials"),
        ("PIN", "", "credentials"),
        ("Verification code", "", "credentials"),
        ("Card no.", "", "credentials"),
    ],
)
def test_detector_classifies_categories(label, url, category):
    assert CriticalDetector().classify(label, url) == category


@pytest.mark.parametrize("label,secret", [
    ("Enter PIN", True), ("PIN code", False), ("Zip/PIN code", False), ("Exp. month", True), ("MM / YY", True),
    ("Expiration date", True), ("Expired listings", False), ("Link expires", False), ("cardnumber", True),
    ("DD/MM/YYYY", False), ("Date of birth (DD / MM / YYYY)", False), ("Card expiry", True),
    ("Expiration", True), ("Card PIN code", True), ("Enter your 4-digit PIN code", True), ("card_number", True),
    ("exp_month", True), ("cvv_code", True), ("Export date", False),
    # bare MM/YYYY is a date format, not a card field ("Start date (MM/YYYY)"); "Expiry (MM/YYYY)" matches by expiry
    ("Start date (MM/YYYY)", False), ("Expiry (MM/YYYY)", True), ("cc_number", True), ("ccnum", True),
    ("creditCardNumber", True), ("exp_yy", True), ("cc-exp", True), ("Valid thru", True), ("CSC", True),
    ("Card #", True), ("Card verification value", True), ("PIN / ZIP code", False), ("CC: alice@x.org", False),
])
def test_credential_stems_need_context(label, secret):
    assert CriticalDetector().credential(label) is secret


def test_credential_check_ignores_category_order():
    d = CriticalDetector()
    label = "Enter your password to delete your account"
    assert d.classify(label) == "destructive" and d.credential(label)
    assert not d.credential("Pinterest") and not d.credential("Spinner") and not d.credential("Delete")


def test_detector_ignores_plain_navigation():
    labels = ["Home", "Next page", "Search", "Sort by price", "View cart", "Back to results", "Displays", "Passage",
              "Hotpot", "Lessn"]
    assert [CriticalDetector().classify(x) for x in labels] == [None] * len(labels)


def test_critical_override_stops_payment():
    g = CriticalOverride(FixedThreshold(0.5), CriticalDetector(), {"a": "Place order", "b": "Back"})
    assert g.decide({"a": 0.9, "b": 0.1}) == GateDecision("escalate", None, ["a"], "stop:payment")
    g.set_labels({"a": "Place order", "b": "Back"})
    assert g.decide({"a": 0.1, "b": 0.9}).action == "act"


def test_critical_override_classifies_chosen_option_not_argmax():
    inner = SimpleNamespace(name="stub", decide=lambda p: GateDecision("act", "b", ["b"], "stub"))
    g = CriticalOverride(inner, CriticalDetector(), {"a": "Pay", "b": "Back"})
    assert g.decide({"a": 0.9, "b": 0.1}).action == "act"
    g.set_labels({"a": "Back", "b": "Pay"})
    assert g.decide({"a": 0.9, "b": 0.1}) == GateDecision("escalate", None, ["b"], "stop:payment")


def test_critical_override_flags_other_categories_via_set_labels():
    g = CriticalOverride(FixedThreshold(0.5), CriticalDetector())
    g.set_labels({"a": "Delete repository"})
    # full list: a one-option confirm prompt would bias the label the adaptive gate learns from
    assert g.decide({"a": 0.9, "b": 0.1}) == GateDecision("escalate", None, [], "critical:destructive")


@pytest.mark.parametrize("label", ["Delete", "Pay"])
def test_critical_override_passes_escalations_through(label):
    for inner, probs in ((FixedThreshold(0.5), {"a": 0.4, "b": 0.3, "c": 0.3}), (SplitConformal(0.1), {"b": 0.3, "a": 0.7})):
        g = CriticalOverride(inner, CriticalDetector(), {"a": label})
        assert g.decide(probs) == inner.decide(probs)


def test_critical_override_labels_reset_after_decide():
    g = CriticalOverride(FixedThreshold(0.5), CriticalDetector(), {"a": "Pay"})
    assert g.decide({"a": 0.9, "b": 0.1}).reason == "stop:payment"
    assert g.decide({"a": 0.9, "b": 0.1}).action == "act"


def test_critical_override_delegates_fit_and_update():
    inner = AdaptiveConformal(0.1)
    g = CriticalOverride(inner, CriticalDetector())
    g.fit([{"a": 0.9, "b": 0.1}] * 20, ["a"] * 20)
    g.update({"a": 0.9, "b": 0.1}, "b")
    assert len(inner.scores) == 20 and inner.alpha < 0.1


@pytest.mark.parametrize("name", ["passCode", "PassCode", "passWord", "PassWord", "CVVCode", "OTPCode", "SSNLast4",
                                  "cardNum", "card_num", "ccnum", "creditCardNumber"])
def test_credential_dom_names_read_as_written_and_as_words(name):
    assert CriticalDetector().credential(name)


@pytest.mark.parametrize("label,cat", [("UnSubscribe", "destructive"), ("postCode", None), ("zipCode", None),
                                       ("deleteButton", "destructive"), ("submitOrder", "payment"),
                                       ("Post comment", "send"), ("postReply", "send")])
def test_camel_case_split_does_not_change_categories_read_as_written(label, cat):
    assert CriticalDetector().classify(label) == cat
