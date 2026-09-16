"""Dollar value of a player's wins to his own program."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from athletevalue.economics.revenue import RevenueModel
from athletevalue.economics.tournament import BidModel

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ProgramContext:
    team: str
    games: int
    wins: int
    power: bool
    revenue_base: float
    """Recent average reported men's basketball revenue, USD."""
    conference_members: int


@dataclass(frozen=True)
class ProgramValueDraws:
    win_revenue: FloatArray
    """Revenue from the player's wins, this season and next, through the fitted effect."""
    bid_revenue: FloatArray
    """Revenue from the change in bid probability, through the fitted bid effect."""
    tournament_units: FloatArray
    """The school's share of conference tournament-unit money from the bid change."""
    bid_probability_change: FloatArray

    @property
    def total(self) -> FloatArray:
        result: FloatArray = self.win_revenue + self.bid_revenue + self.tournament_units
        return result

    @property
    def estimated_part(self) -> FloatArray:
        """Everything except the tournament-unit share, which rests on a user assumption."""
        result: FloatArray = self.win_revenue + self.bid_revenue
        return result


def program_value_draws(
    war: FloatArray,
    context: ProgramContext,
    revenue: RevenueModel,
    bids: BidModel,
    *,
    unit_value: float,
    units_per_bid: float,
    share_multiplier: float,
    rng: np.random.Generator,
) -> ProgramValueDraws:
    """Pair each WAR draw with a bootstrap replicate of the revenue model."""
    n = war.size
    pick = rng.integers(0, revenue.draws.shape[0], n)
    win_effect = revenue.win_effect()[pick]
    bid_effect = revenue.bid_effect()[pick]

    win_pct = context.wins / context.games
    without = np.clip(win_pct - war / context.games, 0.0, 1.0)
    delta_p = bids.probability(win_pct, context.power) - bids.probability(without, context.power)

    win_revenue = war * win_effect * context.revenue_base
    bid_revenue = delta_p * bid_effect * context.revenue_base
    school_share = share_multiplier / context.conference_members
    units = delta_p * units_per_bid * unit_value * school_share
    return ProgramValueDraws(
        win_revenue=win_revenue,
        bid_revenue=bid_revenue,
        tournament_units=units,
        bid_probability_change=delta_p,
    )
