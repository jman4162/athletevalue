"""The valuation a user sees."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict

from athletevalue.schemas.estimate import Estimate
from athletevalue.schemas.identity import PlayerRef
from athletevalue.schemas.source import SourceReference


class Driver(BaseModel):
    model_config = ConfigDict(frozen=True)

    sign: str
    """``+`` raises value or price, ``-`` lowers it, ``~`` is context."""
    text: str


class PlayerValuation(BaseModel):
    """Separate estimates of impact, value to the program and market price for one player-season."""

    model_config = ConfigDict(frozen=True)

    player: PlayerRef
    conference: str
    role: str | None
    athletic_impact: Estimate
    offense: Estimate
    defense: Estimate
    war: Estimate
    war_linear: Estimate
    replacement_definition: str
    replacement_level: float
    war_sensitivity: dict[str, float]
    """Median WAR under each replacement definition, keyed by definition name."""
    program_value: Estimate | None
    """Revenue this season from the player's wins and bid, under EADA accounting."""
    program_value_two_season: Estimate | None = None
    """The same plus next season's carry-over, which accrues whether or not he stays."""
    program_value_components: dict[str, Estimate]
    roster_market_value: Estimate | None
    allocated_market_value: Estimate | None = None
    """The role-and-impact allocation, kept when the fitted model sets the market value."""
    observed_price: Estimate | None
    price_basis: str | None
    surplus: Estimate | None
    quadrant: str | None
    drivers: list[Driver]
    model_version: str
    as_of: date
    data_through: date
    sources: list[SourceReference]
    assumptions_used: list[str]
    warnings: list[str]

    def summary(self) -> str:
        from athletevalue.valuation.report import render_summary

        return render_summary(self)
