"""The program-economics models, assembled from already-loaded frames."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from athletevalue.assumptions.registry import AssumptionRegistry
from athletevalue.economics.panel import revenue_base, revenue_panel
from athletevalue.economics.program_value import ProgramContext
from athletevalue.economics.revenue import RevenueModel, fit_revenue_model
from athletevalue.economics.tournament import (
    BidModel,
    fit_bid_model,
    holdout_bid_predictions,
    marginal_units_per_bid,
    units_per_bid,
)
from athletevalue.schemas.source import SourceReference


@dataclass(frozen=True)
class EconomicsFrames:
    outcomes: pl.DataFrame
    """team, season, conference, games, wins, ncaa_bid, ncaa_games, ncaa_units, sos
    for every season."""
    eada: pl.DataFrame
    crosswalk: pl.DataFrame
    sources: tuple[SourceReference, ...] = field(default=())
    notes: tuple[str, ...] = field(default=())
    """Seasons skipped because a source has not published them, and similar."""


@dataclass(frozen=True)
class EconomicsModel:
    panel: pl.DataFrame
    outcomes: pl.DataFrame
    revenue: RevenueModel
    bids: BidModel
    bid_holdout: pl.DataFrame
    """team, season, predicted, observed, each season scored by a fit without it."""
    units_per_bid: float
    """Units a bid in doubt earns, which is what a player's wins can add or lose."""
    units_per_bid_field: float
    """Units the average bid earns, deep runs included; reported for contrast only."""
    power_conferences: frozenset[str]
    reference_seasons: int
    sources: tuple[SourceReference, ...]
    notes: tuple[str, ...] = ()

    def context(
        self,
        team: str,
        *,
        season: int,
        games: int,
        wins: int,
        conference: str,
        members: int,
        sos: float,
    ) -> ProgramContext | None:
        """Program context, or ``None`` when EADA has no revenue for the school."""
        base = revenue_base(
            self.panel, team, through_season=season, n_seasons=self.reference_seasons
        )
        if base is None:
            return None
        return ProgramContext(
            team=team,
            games=games,
            wins=wins,
            power=conference in self.power_conferences,
            revenue_base=base,
            conference_members=members,
            sos=sos,
        )


def assemble_economics(
    frames: EconomicsFrames, registry: AssumptionRegistry, *, seed: int = 0
) -> EconomicsModel:
    excluded = frozenset(int(s) for s in registry.get("economics.excluded_seasons").numbers())
    power = frozenset(registry.get("market.power_conferences").names())
    first = int(registry.get("economics.panel_first_season").scalar())
    panel = revenue_panel(
        frames.outcomes.filter(pl.col("season") >= first - 1), frames.eada, frames.crosswalk
    )
    revenue = fit_revenue_model(
        panel.filter(pl.col("season") >= first),
        excluded_seasons=excluded,
        min_seasons=int(registry.get("economics.panel_min_seasons").scalar()),
        n_boot=int(registry.get("economics.bootstrap_draws").scalar()),
        rng=np.random.default_rng(seed),
    )
    ridge = registry.get("economics.bid_ridge_lambda").scalar()
    bids = fit_bid_model(
        frames.outcomes, power_conferences=power, excluded_seasons=excluded, ridge=ridge
    )
    return EconomicsModel(
        panel=panel,
        outcomes=frames.outcomes,
        revenue=revenue,
        bids=bids,
        bid_holdout=holdout_bid_predictions(
            frames.outcomes, power_conferences=power, excluded_seasons=excluded, ridge=ridge
        ),
        units_per_bid=marginal_units_per_bid(
            frames.outcomes, bids, power_conferences=power, excluded_seasons=excluded
        ),
        units_per_bid_field=units_per_bid(frames.outcomes, excluded_seasons=excluded),
        power_conferences=power,
        reference_seasons=int(registry.get("economics.revenue_reference_seasons").scalar()),
        sources=frames.sources,
        notes=frames.notes,
    )
