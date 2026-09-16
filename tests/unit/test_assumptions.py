from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from athletevalue.assumptions.models import Assumption, AssumptionBasis
from athletevalue.assumptions.registry import AssumptionRegistry, UnknownAssumptionError
from athletevalue.schemas.evidence import EvidenceStatus


def test_registry_loads_and_statuses_follow_basis(registry):
    assert len(registry) > 30
    assert registry.get("mbb.impact.ridge_lambda").status is EvidenceStatus.ESTIMATED
    assert registry.get("market.rotation_size").status is EvidenceStatus.SCENARIO
    assert registry.get("mbb.wins.pythag_exponent").status is EvidenceStatus.DERIVED


def test_unknown_assumption_is_an_error(registry):
    with pytest.raises(UnknownAssumptionError):
        registry.get("mbb.made.up")


def test_published_basis_needs_citation():
    with pytest.raises(ValidationError):
        Assumption(
            assumption_id="x",
            description="d",
            unit="u",
            basis=AssumptionBasis.PUBLISHED_THIRD_PARTY,
            value=1.0,
        )


def test_modelling_choice_needs_rationale():
    with pytest.raises(ValidationError):
        Assumption(
            assumption_id="x",
            description="d",
            unit="u",
            basis=AssumptionBasis.MODELLING_CHOICE,
            value=1.0,
        )


def test_derived_basis_must_not_pin_value():
    with pytest.raises(ValidationError):
        Assumption(
            assumption_id="x",
            description="d",
            unit="u",
            basis=AssumptionBasis.DERIVED_FROM_DATA,
            value=1.0,
        )


def test_user_file_overrides_packaged_entry(tmp_path: Path):
    override = tmp_path / "what_if.toml"
    override.write_text(
        '["market.performance_exponent"]\n'
        'description = "flatter pay"\nunit = "dimensionless"\nbasis = "user_input"\nvalue = 0.0\n',
        encoding="utf-8",
    )
    loaded = AssumptionRegistry.load(extra_paths=(override,))
    assert loaded.get("market.performance_exponent").scalar() == 0.0


def test_typed_accessors(registry):
    assert registry.get("market.roster_budget_power_2026").interval() == (7_000_000.0, 10_000_000.0)
    assert "SEC" in registry.get("market.power_conferences").names()
    assert registry.get("mbb.impact.drop_garbage_time").flag() is True
    with pytest.raises(TypeError):
        registry.get("mbb.impact.lambda_grid").scalar()
