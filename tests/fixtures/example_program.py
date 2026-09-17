"""The synthetic program behind the README example and the roster figure.

A synthetic season with a one-season box prior, and an economics model with fixed
revenue and bid effects, so every layer runs without downloads.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from athletevalue.assumptions.registry import AssumptionRegistry, default_registry
from athletevalue.economics.revenue import RevenueModel
from athletevalue.economics.tournament import BidModel
from athletevalue.frames.box import player_box_totals
from athletevalue.impact.prior import fit_box_prior
from athletevalue.valuation.economy import EconomicsModel
from athletevalue.valuation.season import SeasonModel, assemble_season, baseline_rapm
from tests.fixtures.raw_season import make_raw_season


def example_program(
    registry: AssumptionRegistry | None = None,
) -> tuple[SeasonModel, EconomicsModel, AssumptionRegistry]:
    registry = registry or default_registry()
    training, _ = make_raw_season(seed=1, season=2025)
    prior = fit_box_prior(
        [(2025, baseline_rapm(training, registry).table, player_box_totals(training.player_box))],
        pseudo_possessions=registry.get("mbb.prior.rate_pseudo_possessions").scalar(),
        ridge=registry.get("mbb.prior.box_ridge").scalar(),
    )
    frames, _ = make_raw_season(seed=0, season=2026)
    season = assemble_season(frames, registry, prior=prior)
    teams = season.teams["team"].to_list()
    economics = EconomicsModel(
        panel=pl.DataFrame(
            {"team": teams, "season": 2025, "rev_men": [12e6 if "Big" in t else 2e6 for t in teams]}
        ),
        outcomes=pl.DataFrame(),
        revenue=RevenueModel(
            point=np.zeros(4),
            draws=np.tile([0.004, 0.002, 0.03, 0.02], (20, 1)),
            n_obs=1,
            n_schools=1,
            first_season=2012,
            last_season=2025,
        ),
        bids=BidModel(coef=np.array([-10.0, 15.0, -6.0, 14.0]), n_obs=1),
        units_per_bid=1.9,
        power_conferences=frozenset({"Big Ten"}),
        reference_seasons=3,
        sources=(),
    )
    return season, economics, registry


def example_player(season: SeasonModel) -> dict[str, object]:
    """The README example: the highest-rated player with a rating of their own."""
    return season.rapm.table.filter(~pl.col("pooled")).row(0, named=True)
