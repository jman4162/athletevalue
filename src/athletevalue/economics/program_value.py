"""Dollar value of a player's wins to the player's own program."""

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
    """Revenue attributed to a player's wins, split by when it arrives.

    ``annual`` is what the season itself brings: this season's win and bid effects
    plus the school's share of the tournament units the bid earns. ``two_season``
    adds next season's carry-over, which accrues whether or not the player is still
    on the roster and so should not be set against one season of pay.
    """

    win_revenue_current: FloatArray
    win_revenue_next: FloatArray
    bid_revenue_current: FloatArray
    bid_revenue_next: FloatArray
    tournament_units: FloatArray
    """The school's share of conference tournament-unit money from the bid change."""
    bid_probability_change: FloatArray

    @property
    def annual(self) -> FloatArray:
        result: FloatArray = (
            self.win_revenue_current + self.bid_revenue_current + self.tournament_units
        )
        return result

    @property
    def two_season(self) -> FloatArray:
        result: FloatArray = self.annual + self.win_revenue_next + self.bid_revenue_next
        return result

    @property
    def total(self) -> FloatArray:
        """Two-season total, kept for callers that predate the annual split."""
        return self.two_season

    @property
    def win_revenue(self) -> FloatArray:
        result: FloatArray = self.win_revenue_current + self.win_revenue_next
        return result

    @property
    def bid_revenue(self) -> FloatArray:
        result: FloatArray = self.bid_revenue_current + self.bid_revenue_next
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

    win_pct = context.wins / context.games
    without = np.clip(win_pct - war / context.games, 0.0, 1.0)
    delta_p = bids.probability(win_pct, context.power) - bids.probability(without, context.power)

    school_share = share_multiplier / context.conference_members
    return ProgramValueDraws(
        win_revenue_current=war * revenue.win_effect_current()[pick] * context.revenue_base,
        win_revenue_next=war * revenue.win_effect_next()[pick] * context.revenue_base,
        bid_revenue_current=delta_p * revenue.bid_effect_current()[pick] * context.revenue_base,
        bid_revenue_next=delta_p * revenue.bid_effect_next()[pick] * context.revenue_base,
        tournament_units=delta_p * units_per_bid * unit_value * school_share,
        bid_probability_change=delta_p,
    )
