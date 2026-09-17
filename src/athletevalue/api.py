"""Public entry points that load data and run the models."""

from __future__ import annotations

import polars as pl

from athletevalue.assumptions.registry import AssumptionRegistry, default_registry
from athletevalue.identity.resolve import resolve_team
from athletevalue.market.registry import load_deal_registry
from athletevalue.sources.cache import ArtifactCache
from athletevalue.sources.sportsdataverse import LATEST_SEASON, SdvClient, SdvDataset
from athletevalue.sources.torvik import TorvikClient
from athletevalue.sports.mbb.economics_loaders import load_economics_frames
from athletevalue.sports.mbb.loaders import load_box_prior, load_season_frames
from athletevalue.valuation.economy import EconomicsModel, assemble_economics
from athletevalue.valuation.player import team_table
from athletevalue.valuation.player import value_player as compose_player
from athletevalue.valuation.result import PlayerValuation
from athletevalue.valuation.season import SeasonModel, assemble_season
from athletevalue.valuation.team import team_draws
from athletevalue.valuation.validate import ValidationReport, validate_season


def fit_season(
    season: int,
    *,
    cache: ArtifactCache | None = None,
    registry: AssumptionRegistry | None = None,
    lam: float | None = None,
    choose_lambda_by_cv: bool = False,
    prior: bool | None = None,
    last_available_season: int | None = None,
) -> SeasonModel:
    """Download (or reuse) one season's inputs and fit ratings, team ratings and win models.

    *prior* defaults to the registry's ``mbb.prior.use_box_prior``. The prior is fit on
    other seasons up to *last_available_season* (default: *season* or the latest season
    with data, whichever is later).
    """
    cache = cache or ArtifactCache.default()
    registry = registry or default_registry()
    frames = load_season_frames(season, cache)
    use_prior = registry.get("mbb.prior.use_box_prior").flag() if prior is None else prior
    box_prior = (
        load_box_prior(
            season,
            cache,
            registry,
            last_season=max(season, last_available_season or LATEST_SEASON),
        )
        if use_prior
        else None
    )
    return assemble_season(
        frames, registry, lam=lam, choose_lambda_by_cv=choose_lambda_by_cv, prior=box_prior
    )


def validate(
    model: SeasonModel,
    *,
    cache: ArtifactCache | None = None,
    registry: AssumptionRegistry | None = None,
    use_torvik: bool = True,
) -> ValidationReport:
    """Compare a fitted season with the SportsDataverse RAPM and Torvik team ratings."""
    cache = cache or ArtifactCache.default()
    registry = registry or default_registry()
    reference = SdvClient(cache).frame(SdvDataset.REFERENCE_RAPM, model.season)
    torvik = TorvikClient(cache).team_results(model.season) if use_torvik else None
    return validate_season(model, registry, reference_rapm=reference, torvik=torvik)


def fit_economics(
    last_season: int,
    *,
    cache: ArtifactCache | None = None,
    registry: AssumptionRegistry | None = None,
) -> EconomicsModel:
    """Fit the revenue and tournament-bid models on every season through *last_season*."""
    cache = cache or ArtifactCache.default()
    registry = registry or default_registry()
    return assemble_economics(load_economics_frames(cache, last_season=last_season), registry)


def value_player(
    name: str,
    season: int,
    *,
    team: str | None = None,
    sport: str = "mbb",
    cache: ArtifactCache | None = None,
    registry: AssumptionRegistry | None = None,
    economics: bool = True,
    seed: int = 0,
) -> PlayerValuation:
    """Value one player-season: impact, wins, program value, market price and surplus."""
    if sport != "mbb":
        raise ValueError(f"sport {sport!r} is not modelled yet; v0.1 covers men's basketball")
    cache = cache or ArtifactCache.default()
    registry = registry or default_registry()
    model = fit_season(season, cache=cache, registry=registry)
    econ = fit_economics(season, cache=cache, registry=registry) if economics else None
    return compose_player(
        model, econ, registry, name, team=team, deals=load_deal_registry(), seed=seed
    )


def value_team(
    team: str,
    season: int,
    *,
    cache: ArtifactCache | None = None,
    registry: AssumptionRegistry | None = None,
    seed: int = 0,
) -> pl.DataFrame:
    """Value-vs-price table for every player on *team*."""
    cache = cache or ArtifactCache.default()
    registry = registry or default_registry()
    model = fit_season(season, cache=cache, registry=registry)
    econ = fit_economics(season, cache=cache, registry=registry)
    resolved = resolve_team(model.teams, team)
    return team_table(team_draws(model, econ, registry, resolved, seed=seed))


__all__ = [
    "fit_economics",
    "fit_season",
    "load_economics_frames",
    "validate",
    "value_player",
    "value_team",
]
