from __future__ import annotations

import itertools

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from athletevalue.market.allocation import AllocationRules, allocate, assign_roles
from athletevalue.market.budget import BudgetTier, MarketUnavailableError, roster_budget
from athletevalue.market.registry import load_deal_registry, observed_annual_pay
from athletevalue.market.surplus import Quadrant, quadrant
from athletevalue.schemas.evidence import EvidenceStatus

RULES = AllocationRules(
    replacement=-2.0,
    rotation_size=3,
    min_possession_share=0.05,
    performance_exponent=1.0,
    performance_floor=0.5,
    rotation_ratio=(0.8, 0.8),
    bench_ratio=(0.5, 0.5),
)


def _roster(nets, shares, se=0.0):
    return pl.DataFrame(
        {
            "athlete_id": [f"p{i}" for i in range(len(nets))],
            "net": nets,
            "se_net": [se] * len(nets),
            "possession_share": shares,
        }
    )


def test_roles_follow_possession_rank():
    shares = np.array([0.8, 0.7, 0.6, 0.55, 0.5, 0.4, 0.3, 0.25, 0.2, 0.1, 0.01])
    roles = assign_roles(shares, RULES)
    assert roles[:5] == ["starter"] * 5
    assert roles[5:8] == ["rotation"] * 3
    assert roles[8:10] == ["bench"] * 2
    assert roles[10] == "unpaid"


def test_equal_impact_reproduces_role_ratios():
    roster = _roster([3.0] * 10, [0.8, 0.7, 0.6, 0.55, 0.5, 0.4, 0.3, 0.25, 0.2, 0.1])
    result = allocate(roster, np.full(50, 1e6), RULES, rng=np.random.default_rng(0))
    pay = result.pay.mean(axis=0)
    assert pay[5] / pay[0] == pytest.approx(0.8)
    assert pay[8] / pay[0] == pytest.approx(0.5)
    np.testing.assert_allclose(result.pay.sum(axis=1), 1e6)


@settings(max_examples=40, deadline=None)
@given(
    nets=st.lists(st.floats(-15, 20), min_size=6, max_size=15),
    budget=st.floats(1e5, 2e7),
)
def test_allocation_spends_the_budget_and_is_monotone_within_role(nets, budget):
    n = len(nets)
    shares = np.linspace(0.8, 0.06, n)
    result = allocate(
        _roster(nets, shares), np.full(3, budget), RULES, rng=np.random.default_rng(0)
    )
    np.testing.assert_allclose(result.pay.sum(axis=1), budget, rtol=1e-9)
    assert np.all(result.pay >= 0)
    starters = [i for i, r in enumerate(result.roles) if r == "starter"]
    ordered = sorted(starters, key=lambda i: max(nets[i] + 2.0, 0.5))
    pay = result.pay[0]
    assert all(pay[a] <= pay[b] + 1e-6 for a, b in itertools.pairwise(ordered))


def test_budget_tiers_and_missing_season(registry):
    rng = np.random.default_rng(0)
    power = roster_budget("SEC", 2026, registry, n_draws=20000, rng=rng)
    assert power.tier is BudgetTier.POWER and power.status is EvidenceStatus.ESTIMATED
    low, high = np.quantile(power.draws, [0.1, 0.9])
    assert low == pytest.approx(7e6, rel=0.03) and high == pytest.approx(10e6, rel=0.03)
    other = roster_budget("MEAC", 2026, registry, n_draws=10, rng=rng)
    assert other.tier is BudgetTier.TIER_C and other.status is EvidenceStatus.SCENARIO
    with pytest.raises(MarketUnavailableError):
        roster_budget("SEC", 2025, registry, n_draws=10, rng=rng)


def test_packaged_registry_is_valid_and_lookup_works():
    assert load_deal_registry() == []
    frame = pl.DataFrame(
        {
            "deal_id": ["d1", "d2"],
            "athlete_name": ["Jane Doe", "Jane Doe"],
            "season": [2026, 2026],
            "school": ["Duke", "Duke"],
            "sport": ["mbb", "mbb"],
            "cash_value": [100000.0, 60000.0],
            "duration_months": [12.0, 6.0],
            "deal_type": ["revenue_share", "endorsement"],
            "source_url": ["https://example.org/a", "https://example.org/b"],
            "source_quality": ["named_report", "named_report"],
        }
    )
    records = load_deal_registry(frame)
    total, used = observed_annual_pay(records, name="jane doe", school="Duke", season=2026)
    assert total == pytest.approx(100000 + 120000) and len(used) == 2
    assert observed_annual_pay(records, name="Jane Doe", school="Duke", season=2025) is None


def test_quadrants():
    assert quadrant(10, 1, value_cut=5, price_cut=5) is Quadrant.UNDERVALUED
    assert quadrant(10, 10, value_cut=5, price_cut=5) is Quadrant.STAR
    assert quadrant(1, 10, value_cut=5, price_cut=5) is Quadrant.OVERVALUED
    assert quadrant(1, 1, value_cut=5, price_cut=5) is Quadrant.LOW_PRIORITY
