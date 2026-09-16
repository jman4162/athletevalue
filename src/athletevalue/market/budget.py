"""Reported men's basketball roster budgets by conference tier."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from athletevalue.assumptions.registry import AssumptionRegistry
from athletevalue.schemas.evidence import EvidenceStatus
from athletevalue.uncertainty.draws import FloatArray, lognormal_from_range


class MarketUnavailableError(LookupError):
    """Raised for a season with no cited roster-budget figures."""


class BudgetTier(StrEnum):
    POWER = "power"
    TIER_B = "tier_b"
    TIER_C = "tier_c"


@dataclass(frozen=True)
class RosterBudget:
    tier: BudgetTier
    draws: FloatArray
    status: EvidenceStatus
    assumption_ids: tuple[str, ...]


def budget_tier(
    conference: str, registry: AssumptionRegistry
) -> tuple[BudgetTier, tuple[str, ...]]:
    """Tier for *conference* and the registry entries that decided it."""
    if conference in registry.get("market.power_conferences").names():
        return BudgetTier.POWER, ("market.power_conferences",)
    ids = ("market.power_conferences", "market.tier_b_conferences")
    if conference in registry.get("market.tier_b_conferences").names():
        return BudgetTier.TIER_B, ids
    return BudgetTier.TIER_C, ids


def roster_budget(
    conference: str,
    season: int,
    registry: AssumptionRegistry,
    *,
    n_draws: int,
    rng: np.random.Generator,
) -> RosterBudget:
    tier, tier_ids = budget_tier(conference, registry)
    range_id = f"market.roster_budget_{tier.value}_{season}"
    if range_id not in registry:
        # Budgets are season-specific; a season is supported only once a cited range
        # for it is added to the registry.
        raise MarketUnavailableError(f"no cited roster-budget range {range_id!r}")
    level_id = "market.budget_range_level"
    low, high = registry.get(range_id).interval()
    draws = lognormal_from_range(low, high, registry.get(level_id).scalar(), n_draws, rng)
    ids = (*tier_ids, range_id, level_id)
    status = EvidenceStatus.weakest(*(registry.get(i).status for i in ids))
    return RosterBudget(tier=tier, draws=draws, status=status, assumption_ids=ids)
