import re

from surety.gate.policies import GatePolicy
from surety.types import GateDecision

# Stems and compounds, tuned for recall; checked in order, so payment wins over send.
_STEMS = {
    "payment": [
        r"pay\w*",
        r"purchas\w*",
        r"buy\w*",
        r"place[\s-]+(?:your[\s-]+)?order",
        r"check[\s-]?out\b",
        r"(?:submit|complete|confirm)[\s-]+(?:your[\s-]+|my[\s-]+)?(?:order|purchase|payment)",
        r"subscri\w*",
        r"add[\s-]+(?:a[\s-]+)?card",
    ],
    "destructive": [r"delet\w*", r"remov\w*", r"discard\w*", r"clear[\s-]+all", r"eras\w*", r"unsubscri\w*"],
    "send": [
        r"send\w*",
        r"post\b",
        r"publish\w*",
        r"repl(?:y|ies)\b",
        r"submit[\s-]+application",
        r"tweet\w*",
    ],
    "account": [
        r"sign[\s-]?out\b",
        r"log[\s-]?out\b",
        r"change[\s-]+password",
        r"deactivat\w*",
        r"close[\s-]+(?:your[\s-]+)?account",
    ],
    "credentials": [
        r"password",
        r"passcode",
        r"card[\s-]*(?:number\b|no\b|#)",
        r"cc[\s-]*(?:num\w*|exp\w*|csc|cvv|cvc)\b",
        r"cv[vc]\d?\b",
        r"csc\b",
        r"security[\s-]+code",
        r"valid[\s-]+(?:thru|through|until)\b",
        r"pin\b(?![\s/,-]*(?:code|zip))",  # not "PIN code", "PIN / ZIP code" (Indian postcode) ...
        r"(?:card|atm|security|secret|\d[\s-]?digit)[\s-]+pin\b",  # ... unless it is a card/ATM PIN
        r"social[\s-]+security",
        r"ssn\b",
        r"one[\s-]time[\s-]+(?:code|password)",
        r"verification[\s-]+(?:code|value)",
        r"otp\b",
        r"exp(?:iry|iration)\b",  # not "expired", "expires"
        r"exp\.?[\s-]*(?:date|month|year|mm|yy)\b",
        r"(?<!d/)(?<!d /)(?<!d/ )(?<!d / )mm\s*/\s*yy\b",  # card MM/YY; not DD/MM/YY(YY) nor a "Start date (MM/YYYY)"
    ],
}
_RULES = {cat: re.compile(rf"\b(?:{'|'.join(stems)})", re.IGNORECASE) for cat, stems in _STEMS.items()}


def _words(text: str) -> str:
    """DOM names as words: card_number, creditCardNumber -> card number, credit Card Number."""
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text.replace("_", " "))


class CriticalDetector:
    def classify(self, label: str, url: str = "") -> str | None:
        text = _words(f"{label} {url}")
        return next((cat for cat, rule in _RULES.items() if rule.search(text)), None)

    def credential(self, label: str) -> bool:
        """Independent of category order: 'Enter your password to delete your account' classifies as destructive."""
        return bool(_RULES["credentials"].search(_words(label)))


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
        labels, self.labels = self.labels, {}
        if d.action != "act":
            return d
        # unlabelled ids are classified by the id itself: recall over precision
        cat = self.detector.classify(labels.get(d.option_id, d.option_id))
        if cat == "payment":
            return GateDecision("escalate", None, [d.option_id], "stop:payment")
        if cat:
            # the full list, not a confirm prompt: the answer is also the gate's update label
            return GateDecision("escalate", None, [], f"critical:{cat}")
        return d
