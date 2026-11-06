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
        r"confirm[\s-]+(?:your[\s-]+)?order",
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
}
_RULES = {cat: re.compile(rf"\b(?:{'|'.join(stems)})", re.IGNORECASE) for cat, stems in _STEMS.items()}


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
        labels, self.labels = self.labels, {}
        top = max(probs, key=probs.get)
        # unlabelled ids are classified by the id itself: recall over precision
        cat = self.detector.classify(labels.get(top, top))
        if cat == "payment":
            candidates = d.candidate_set if d.action == "escalate" else [top]
            return GateDecision("escalate", None, candidates, "stop:payment")
        if cat and d.action == "act":
            return GateDecision("escalate", None, [d.option_id], f"critical:{cat}")
        return d
