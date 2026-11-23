"""WebGuard (osunlp) critical-detector recall. Labels: SAFE / LOW / HIGH risk per web action.

The official release (osunlp/WebGuard) has no element text; the authors' annotation export
(boyuanzheng010/webguard) is row-aligned with it after dropping BUG rows and carries the target
text, cut to 30 characters, in `annotation_folder` (annot_<first-pass label>_Tgt_<text>_<uuid>).
The first-pass label is discarded; the reviewed label (`Your Review`) is the official one.
"""
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from surety.calib import metrics as m

LEVELS = ("SAFE", "LOW", "HIGH")
_FOLDER = re.compile(r"^annot_[A-Za-z]+_Tgt_(.*)_[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$", re.S)
# attribute keys before a value (key__value), or a multi-word key cut off at the 30-char limit
_ATTR = re.compile(r"(?:^|(?<=[_\]]))(?:aria-label|aria-describedby|parent_node|name|value|INPUT_VALUE|title|alt|placeholder)"
                   r"(?=__)|(?<=[_\]])(?:aria-label|aria-describedby|parent_node|INPUT_VALUE)(?=_?$)|[\[\]_]")


def target_text(folder: str) -> str | None:
    """'annot_HIGH_Tgt_Send_request_<uuid>' -> 'Send request'; attribute keys (aria-label__, name__…__value__) dropped."""
    if not (hit := _FOLDER.match(folder)):
        return None
    return " ".join(_ATTR.sub(" ", hit.group(1)).split())


def load_actions(path: Path) -> list[dict]:
    rows = pq.read_table(path, columns=["url", "tagHead", "annotation_folder", "Your Review", "website"]).to_pylist()
    out = []
    for r in rows:
        level, text = (r["Your Review"] or "").upper(), target_text(r["annotation_folder"])
        if level in LEVELS and text is not None:
            out.append({"label": text, "url": r["url"] or "", "tag": r["tagHead"] or "", "level": level,
                        "site": r["website"]})
    return out


def flags(actions: list[dict], detector, with_url: bool) -> list[str | None]:
    return [detector.classify(a["label"], a["url"] if with_url else "") for a in actions]


def _rate(k: int, n: int) -> str:
    lo, hi = m.wilson(k, n)
    return f"{k / n:.3f} [{lo:.3f}, {hi:.3f}]" if n else "-"


def recall_table(actions: list[dict], detector) -> str:
    lines = ["| input | level | n | flagged (95% CI) | stop:payment | other categories |", "|---|---|---|---|---|---|"]
    for with_url in (False, True):
        cats = flags(actions, detector, with_url)
        for level in reversed(LEVELS):
            got = [c for c, a in zip(cats, actions) if a["level"] == level]
            k = sum(c is not None for c in got)
            other = Counter(c for c in got if c not in (None, "payment"))
            lines.append(f"| {'label+url' if with_url else 'label'} | {level} | {len(got)} | {_rate(k, len(got))} | "
                         f"{sum(c == 'payment' for c in got)} | "
                         f"{', '.join(f'{c} {n}' for c, n in other.most_common()) or '-'} |")
    return "\n".join(lines) + "\n"


# stem families the current rules lack (spec: submit-with-side-effects, account changes; WebGuard has Spanish sites)
CANDIDATES = {
    "submit": r"submit\w*|enviar",
    "sign up / register / create account": r"sign[\s-]?up|regist\w*|create[\s-]+(?:an?[\s-]+|your[\s-]+)?account",
    "save": r"save\b|save[\s-]+changes|guardar",
    "download / install": r"download\w*|descarg\w*|install\w*",
    "spanish pay / buy / delete": r"pagar|comprar|eliminar|borrar",
    "confirm / continue": r"confirm\w*|continu\w*",
}


def candidates_table(actions: list[dict], detector, candidates: dict[str, str] = CANDIDATES) -> str:
    """What each extra stem family would add on top of the current rules (label+url), per risk level."""
    base = [c is not None for c in flags(actions, detector, with_url=True)]
    n = {lv: sum(a["level"] == lv for a in actions) for lv in LEVELS}
    rules = {name: re.compile(rf"\b(?:{rx})", re.IGNORECASE) for name, rx in candidates.items()}
    rules["all above except confirm / continue"] = re.compile(
        rf"\b(?:{'|'.join(rx for k, rx in candidates.items() if k != 'confirm / continue')})", re.IGNORECASE)
    lines = ["| stems | +HIGH recall | +LOW flagged | +SAFE flagged |", "|---|---|---|---|"]
    for name, rule in rules.items():
        new = [not b and bool(rule.search(f"{a['label']} {a['url']}")) for a, b in zip(actions, base)]
        add = {lv: sum(x for x, a in zip(new, actions) if a["level"] == lv) / n[lv] for lv in LEVELS if n[lv]}
        lines.append(f"| {name} | +{add['HIGH']:.3f} | +{add['LOW']:.3f} | +{add['SAFE']:.3f} |")
    return "\n".join(lines) + "\n"


def missed(actions: list[dict], detector, level: str = "HIGH", top: int = 25) -> list[tuple[str, int]]:
    cats = flags(actions, detector, with_url=True)
    return Counter(a["label"].lower() for c, a in zip(cats, actions) if a["level"] == level and c is None
                   ).most_common(top)
