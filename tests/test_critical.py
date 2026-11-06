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
    ],
)
def test_detector_classifies_categories(label, url, category):
    assert CriticalDetector().classify(label, url) == category


def test_detector_ignores_plain_navigation():
    labels = ["Home", "Next page", "Search", "Sort by price", "View cart", "Back to results", "Displays"]
    assert [CriticalDetector().classify(x) for x in labels] == [None] * len(labels)


def test_critical_override_stops_payment():
    g = CriticalOverride(FixedThreshold(0.5), CriticalDetector(), {"a": "Place order", "b": "Back"})
    assert g.decide({"a": 0.9, "b": 0.1}) == GateDecision("escalate", None, ["a"], "stop:payment")
    g.set_labels({"a": "Place order", "b": "Back"})
    assert g.decide({"a": 0.1, "b": 0.9}).action == "act"


def test_critical_override_stops_payment_when_inner_escalates():
    g = CriticalOverride(FixedThreshold(0.5), CriticalDetector(), {"a": "Pay"})
    assert g.decide({"a": 0.4, "b": 0.3, "c": 0.3}) == GateDecision("escalate", None, [], "stop:payment")
    g = CriticalOverride(SplitConformal(0.1), CriticalDetector(), {"a": "Pay"})
    assert g.decide({"b": 0.3, "a": 0.7}) == GateDecision("escalate", None, ["a", "b"], "stop:payment")


def test_critical_override_flags_other_categories_via_set_labels():
    g = CriticalOverride(FixedThreshold(0.5), CriticalDetector())
    g.set_labels({"a": "Delete repository"})
    assert g.decide({"a": 0.9, "b": 0.1}) == GateDecision("escalate", None, ["a"], "critical:destructive")


def test_critical_override_passes_through_non_payment_escalations():
    inner = FixedThreshold(0.5)
    g = CriticalOverride(inner, CriticalDetector(), {"a": "Delete"})
    probs = {"a": 0.4, "b": 0.3, "c": 0.3}
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
