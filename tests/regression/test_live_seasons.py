"""Full-season checks against published references. Run with ``pytest -m network``."""

from __future__ import annotations

import pytest

from athletevalue import api

pytestmark = pytest.mark.network


@pytest.fixture(scope="module", params=[2025, 2026])
def season_model(request):
    return api.fit_season(request.param)


def test_season_passes_every_gate(season_model):
    report = api.validate(season_model)
    failed = [g for g in report.gates if not g.passed]
    assert not failed, failed


def test_known_player_valuation_is_complete():
    valuation = api.value_player("Yaxel Lendeborg", 2026)
    assert valuation.player.team == "Michigan"
    assert valuation.athletic_impact.value > 5
    assert valuation.war.lower > 0
    assert valuation.program_value is not None and valuation.program_value.value > 0
    assert valuation.roster_market_value is not None
    assert valuation.surplus is not None
    assert "Michigan" in valuation.summary()


def test_team_allocation_spends_power_tier_budget():
    table = api.value_team("Duke", 2026)
    assert 6e6 < table["price"].sum() < 11e6
