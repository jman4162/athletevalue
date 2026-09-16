"""Epistemic status of a number.

Reported, derived, estimated and scenario values are never silently mixed. Every
calculation takes the status of its weakest input, so a dollar figure that
depends on a user-chosen pay split is labelled a scenario even when every other
input was measured.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self


class EvidenceStatus(StrEnum):
    """How much weight a number carries. Listed from strongest to weakest."""

    REPORTED = "reported"
    """Stated by a primary source: a box score, a settlement term, a filing."""

    DERIVED = "derived"
    """Computed from reported data by a documented, deterministic procedure."""

    ESTIMATED = "estimated"
    """Depends on a statistical fit or on a published third-party figure."""

    SCENARIO = "scenario"
    """Depends on a user-chosen assumption; a what-if, not a measurement."""

    UNRESOLVED = "unresolved"
    """Could not be determined from available evidence."""

    @property
    def rank(self) -> int:
        """Position in the strength order; higher is weaker."""
        return _RANK[self]

    @property
    def glyph(self) -> str:
        """Single-character marker used in reports."""
        return _GLYPH[self]

    @classmethod
    def weakest(cls, *statuses: EvidenceStatus) -> Self:
        """Return the weakest of *statuses*; ``REPORTED`` when given none."""
        weakest = max(statuses, key=lambda status: _RANK[status], default=cls.REPORTED)
        return cls(weakest)


_RANK: dict[EvidenceStatus, int] = {
    EvidenceStatus.REPORTED: 0,
    EvidenceStatus.DERIVED: 1,
    EvidenceStatus.ESTIMATED: 2,
    EvidenceStatus.SCENARIO: 3,
    EvidenceStatus.UNRESOLVED: 4,
}

_GLYPH: dict[EvidenceStatus, str] = {
    EvidenceStatus.REPORTED: "●",
    EvidenceStatus.DERIVED: "◆",
    EvidenceStatus.ESTIMATED: "▲",
    EvidenceStatus.SCENARIO: "○",
    EvidenceStatus.UNRESOLVED: "!",
}
