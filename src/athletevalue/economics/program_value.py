"""Dollar value of a player's wins to the player's own program."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from athletevalue.economics.revenue import RevenueModel
from athletevalue.economics.tournament import BidModel

FloatArray = NDArray[np.float64]


class UnitOverlapPolicy(StrEnum):
    """How to keep the unit component and the EADA bid effect off the same dollars.

    Units earned in one tournament are distributed from the following April, so they
    fall in the next fiscal year. The same-season bid effect therefore cannot contain
    unit money and the annual figure never double counts. The next-season bid effect
    can contain the first installment, which only the two-season figure uses.
    """

    NONE = "none"
    """Assume reported revenue excludes unit money; count both in full."""

    NEXT_SEASON_INSTALLMENT = "next_season_installment"
    """Subtract the first installment from the two-season figure, where it can overlap."""

    EXCLUDE_UNITS = "exclude_units"
    """Leave the unit component out of both figures and report it on its own."""


@dataclass(frozen=True)
class UnitTerms:
    """What a bid's tournament-unit money is worth to one school, in today's dollars.

    ``unit_value`` is the whole payout, spread over ``installments`` annual
    instalments beginning ``first_year`` years after the season, so the present value
    is a fraction of it.
    """

    unit_value: float
    units_per_bid: float
    share_multiplier: float
    discount_rate: float
    installments: int
    first_year: int
    overlap: UnitOverlapPolicy = UnitOverlapPolicy.NEXT_SEASON_INSTALLMENT

    def _year(self, k: int) -> float:
        return float((1.0 + self.discount_rate) ** -k)

    def pv_factor(self) -> float:
        """Present value of the whole payout, as a fraction of its undiscounted total."""
        years = range(self.first_year, self.first_year + self.installments)
        return sum(self._year(k) for k in years) / self.installments

    def first_installment_factor(self) -> float:
        """Present value of the first instalment alone, as a fraction of the total."""
        return self._year(self.first_year) / self.installments

    def next_season_factor(self) -> float:
        """Present value of a dollar of next season's revenue effect."""
        return self._year(1)

    def dollars(self, members: int) -> float:
        """One marginal bid's discounted unit money for one school of a conference."""
        return self.unit_value * self.units_per_bid * self.share_multiplier / members


@dataclass(frozen=True)
class ProgramContext:
    team: str
    games: int
    wins: int
    power: bool
    revenue_base: float
    """Recent average reported men's basketball revenue, USD."""
    conference_members: int
    sos: float
    """Mean opponent win percentage, which the bid model weighs against the record."""


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
    """The school's share of conference tournament-unit money from the bid change,
    discounted over the years it is paid in. Always the full component, whatever
    ``overlap`` leaves in the headline figures."""
    tournament_units_first_installment: FloatArray
    """The part of that money arriving in the next fiscal year, which is the only part
    next season's reported revenue can already contain."""
    bid_probability_change: FloatArray
    overlap: UnitOverlapPolicy = UnitOverlapPolicy.NEXT_SEASON_INSTALLMENT

    @property
    def counted_units(self) -> FloatArray:
        """Unit money the headline figures include."""
        if self.overlap is UnitOverlapPolicy.EXCLUDE_UNITS:
            return np.zeros_like(self.tournament_units)
        return self.tournament_units

    @property
    def annual(self) -> FloatArray:
        result: FloatArray = (
            self.win_revenue_current + self.bid_revenue_current + self.counted_units
        )
        return result

    @property
    def two_season(self) -> FloatArray:
        result: FloatArray = self.annual + self.win_revenue_next + self.bid_revenue_next
        if self.overlap is UnitOverlapPolicy.NEXT_SEASON_INSTALLMENT:
            result = result - self.tournament_units_first_installment
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
    units: UnitTerms,
    rng: np.random.Generator,
) -> ProgramValueDraws:
    """Pair each WAR draw with a bootstrap replicate of the revenue model.

    Money that arrives after the season is discounted to the season being valued:
    next season's revenue effects by one year, and the unit instalments over the
    years they are paid in.
    """
    n = war.size
    pick = rng.integers(0, revenue.draws.shape[0], n)

    win_pct = context.wins / context.games
    without = np.clip(win_pct - war / context.games, 0.0, 1.0)
    delta_p = bids.probability(win_pct, context.power, context.sos) - bids.probability(
        without, context.power, context.sos
    )

    next_season = units.next_season_factor()
    undiscounted_units = delta_p * units.dollars(context.conference_members)
    return ProgramValueDraws(
        win_revenue_current=war * revenue.win_effect_current()[pick] * context.revenue_base,
        win_revenue_next=(
            war * revenue.win_effect_next()[pick] * context.revenue_base * next_season
        ),
        bid_revenue_current=delta_p * revenue.bid_effect_current()[pick] * context.revenue_base,
        bid_revenue_next=(
            delta_p * revenue.bid_effect_next()[pick] * context.revenue_base * next_season
        ),
        tournament_units=undiscounted_units * units.pv_factor(),
        tournament_units_first_installment=(undiscounted_units * units.first_installment_factor()),
        bid_probability_change=delta_p,
        overlap=units.overlap,
    )
