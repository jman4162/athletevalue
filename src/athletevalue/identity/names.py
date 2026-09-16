"""Name normalization and matching for players and teams.

Different sources spell the same school differently ("Michigan St." vs "Michigan
State University"). Normalization expands common abbreviations and drops words
that carry no identity; matching then compares token sets.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

_ABBREVIATIONS = {
    "st": "state",
    "fla": "florida",
    "ga": "georgia",
    "ky": "kentucky",
    "mo": "missouri",
    "tenn": "tennessee",
    "ark": "arkansas",
    "colo": "colorado",
    "wash": "washington",
    "ill": "illinois",
    "miss": "mississippi",
    "ala": "alabama",
    "la": "louisiana",
    "mich": "michigan",
    "ind": "indiana",
    "ariz": "arizona",
    "calif": "california",
    "conn": "connecticut",
    "okla": "oklahoma",
    "tex": "texas",
    "va": "virginia",
    "caro": "carolina",
    "so": "southern",
    "ut": "utah",
    "col": "college",
    "univ": "university",
    "mt": "mount",
    "intl": "international",
    "u": "university",
}
_DROP = {"university", "of", "the", "at", "and", "college", "main", "campus", "in"}


def ascii_fold(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def name_tokens(name: str) -> tuple[str, ...]:
    """Lower-case tokens with abbreviations expanded and filler words removed."""
    text = ascii_fold(name).lower().replace("&", " and ")
    text = re.sub(r"[()'.,\-/]", " ", text)
    tokens: list[str] = []
    for raw in text.split():
        if raw == "saint" or (raw == "st" and not tokens):
            tokens.append("saint")
            continue
        token = _ABBREVIATIONS.get(raw, raw)
        if token not in _DROP:
            tokens.append(token)
    return tuple(tokens)


def normalize_name(name: str) -> str:
    return " ".join(name_tokens(name))


def similarity(left: str, right: str) -> float:
    """Blend of token-set overlap and character similarity, in [0, 1]."""
    a, b = set(name_tokens(left)), set(name_tokens(right))
    if not a or not b:
        return 0.0
    jaccard = len(a & b) / len(a | b)
    chars = SequenceMatcher(None, normalize_name(left), normalize_name(right)).ratio()
    return 0.5 * jaccard + 0.5 * chars
