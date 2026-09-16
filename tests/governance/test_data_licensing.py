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
