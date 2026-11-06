import re

from surety.gate.policies import GatePolicy
from surety.types import GateDecision

_KEYWORDS = {
    "payment": "pay|purchase|buy now|place order|checkout|confirm order|subscribe|add card",
    "destructive": "delete|remove|discard|clear all|erase|unsubscribe",
    "send": "send|post|publish|reply|submit application|tweet",
    "account": "sign out|log out|change password|deactivate|close account",
}
_RULES = {
    cat: re.compile(rf"\b(?:{kw.replace(' ', r'\s+')})\b", re.IGNORECASE) for cat, kw in _KEYWORDS.items()
}


class CriticalDetector:
    def classify(self, label: str, url: str = "") -> str | None:
        text = f"{label} {url}"
        return next((cat for cat, rule in _RULES.items() if rule.search(text)), None)


class CriticalOverride:
    def __init__(self, policy: GatePolicy, detector: CriticalDetector, labels: dict[str, str] | None = None):
        self.policy = policy
        self.detector = detector
        self.labels = labels or {}
        self.name = f"{policy.name}+critical"

    def set_labels(self, labels: dict[str, str]) -> None:
        self.labels = labels

    def fit(self, probs: list[dict[str, float]], gold: list[str]) -> None:
        self.policy.fit(probs, gold)

    def update(self, probs: dict[str, float], gold: str, weight: float = 1.0) -> None:
        self.policy.update(probs, gold, weight)

    def decide(self, probs: dict[str, float]) -> GateDecision:
        d = self.policy.decide(probs)
        if d.action != "act":
            return d
        # unlabelled ids are classified by the id itself: recall over precision
        cat = self.detector.classify(self.labels.get(d.option_id, d.option_id))
        if cat is None:
            return d
        reason = "stop:payment" if cat == "payment" else f"critical:{cat}"
        return GateDecision("escalate", None, [d.option_id], reason)
