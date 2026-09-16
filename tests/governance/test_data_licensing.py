"""Restricted data must never reach code that fits shipped values."""

from __future__ import annotations

import re
from pathlib import Path

from athletevalue.assumptions.registry import AssumptionRegistry
from athletevalue.schemas.source import LicenseTag

SRC = Path(__file__).resolve().parents[2] / "src" / "athletevalue"
FITTING_PACKAGES = ("impact", "wins", "economics", "market", "valuation", "sports")
RESTRICTED = re.compile(r"user_csv|kenpom|knight", re.IGNORECASE)


def test_fitting_code_never_mentions_restricted_sources():
    offenders = [
        str(path)
        for package in FITTING_PACKAGES
        for path in (SRC / package).rglob("*.py")
        if RESTRICTED.search(path.read_text(encoding="utf-8"))
    ]
    assert not offenders, offenders


def test_registry_citations_are_redistributable_facts():
    registry = AssumptionRegistry.load()
    for assumption in registry:
        if assumption.citation is not None:
            assert assumption.citation.license is LicenseTag.FACTUAL_CITATION, (
                assumption.assumption_id
            )


def test_every_assumption_is_used_by_code():
    """A registry entry no code reads would mislead a reader about what drives results."""
    code = "\n".join(p.read_text(encoding="utf-8") for p in SRC.rglob("*.py"))
    registry = AssumptionRegistry.load()
    budget_prefix = "market.roster_budget_"
    unused = [
        a.assumption_id
        for a in registry
        if a.assumption_id not in code and not a.assumption_id.startswith(budget_prefix)
    ]
    assert not unused, unused
