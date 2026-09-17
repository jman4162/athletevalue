"""Public entry points that load data and run the models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from athletevalue.assumptions.registry import AssumptionRegistry, default_registry
from athletevalue.identity.resolve import resolve_team
from athletevalue.impact.prior import PriorTrainingError
from athletevalue.market.model import InsufficientLabelsError
from athletevalue.market.registry import load_deal_registry
from athletevalue.schemas.registry import DealRecord
from athletevalue.sources.cache import ArtifactCache, SourceUnavailableError
from athletevalue.sources.sportsdataverse import SdvClient, SdvDataset
from athletevalue.sources.torvik import TorvikClient
from athletevalue.sports.mbb.economics_loaders import load_economics_frames
from athletevalue.sports.mbb.loaders import load_box_prior, load_season_frames
from athletevalue.valuation.economy import EconomicsModel, assemble_economics
from athletevalue.valuation.market_fit import MarketFit, match_labels, player_features
from athletevalue.valuation.market_fit import fit_market as fit_market_labels
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
    seasons before *season* only; *last_available_season* can lower that cap further
    (for example to reproduce what was knowable at an earlier date). The first season
    with data has no earlier season and is fitted without a prior.
    """
    cache = cache or ArtifactCache.default()
    registry = registry or default_registry()
    frames = load_season_frames(season, cache)
    use_prior = registry.get("mbb.prior.use_box_prior").flag() if prior is None else prior
    box_prior = None
    notes: list[str] = []
    if use_prior:
        cap = season - 1
        if last_available_season is not None:
            cap = min(last_available_season, cap)
        try:
            box_prior = load_box_prior(season, cache, registry, last_season=cap)
        except PriorTrainingError as error:
            notes.append(f"no box-score prior: {error}")
    return assemble_season(
        frames,
        registry,
        lam=lam,
        choose_lambda_by_cv=choose_lambda_by_cv,
        prior=box_prior,
        notes=tuple(notes),
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
    torvik = None
    notes: tuple[str, ...] = ()
    if use_torvik:
        try:
            torvik = TorvikClient(cache).team_results(model.season)
        except SourceUnavailableError as error:
            # barttorvik.com refuses some cloud address ranges; the comparison is
            # optional and its absence is reported rather than hidden.
            notes = (f"Torvik comparison skipped: {error}",)
    report = validate_season(model, registry, reference_rapm=reference, torvik=torvik)
    return ValidationReport(season=report.season, gates=report.gates, notes=(*report.notes, *notes))


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


def load_deals(labels: Path | None = None) -> list[DealRecord]:
    """Deals from the packaged registry, plus a user CSV in the same schema if given."""
    deals = load_deal_registry()
    if labels is not None:
        # Read every column as text: numeric ids such as stats.ncaa.org athlete ids would
        # otherwise load as integers and fail the string fields of DealRecord.
        deals += load_deal_registry(pl.read_csv(labels, infer_schema_length=0))
    return deals


def _label_seasons(deals: list[DealRecord], registry: AssumptionRegistry) -> list[int]:
    """Seasons of the deals that can become labels; raises below the registry minimum."""
    types = set(registry.get("market.model.label_deal_types").names())
    labeled = [d for d in deals if d.sport == "mbb" and d.deal_type.value in types]
    minimum = int(registry.get("market.model.min_labels").scalar())
    if not labeled or len(labeled) < minimum:
        raise InsufficientLabelsError(
            f"{len(labeled)} disclosed deals available; at least {minimum} labeled "
            "player-seasons needed"
        )
    return sorted({d.season for d in labeled})


def fit_market(
    deals: list[DealRecord] | None = None,
    *,
    labels: Path | None = None,
    cache: ArtifactCache | None = None,
    registry: AssumptionRegistry | None = None,
    fitted: dict[int, SeasonModel] | None = None,
    economics: EconomicsModel | None = None,
) -> MarketFit:
    """Fit the roster-market model on disclosed men's basketball deals.

    Raises ``InsufficientLabelsError`` when the labels do not meet the registry
    minimums; the valuation then keeps the allocation as the price.
    """
    cache = cache or ArtifactCache.default()
    registry = registry or default_registry()
    deals = load_deals(labels) if deals is None else deals
    seasons = _label_seasons(deals, registry)
    models = dict(fitted or {})
    econ = economics or fit_economics(max(seasons), cache=cache, registry=registry)
    features = []
    for season in seasons:
        if season not in models:
            models[season] = fit_season(season, cache=cache, registry=registry)
        features.append(player_features(models[season], econ, registry))
    table = pl.concat(features, how="vertical_relaxed")
    matched = match_labels(deals, table, registry)
    prices = []
    for season, team in matched.labels.select("season", "team").unique().iter_rows():
        allocation = team_draws(models[season], econ, registry, team, seed=0).allocation
        if allocation is None:
            continue
        medians = np.median(allocation.pay, axis=0)
        prices.append(
            pl.DataFrame(
                {
                    "athlete_id": list(allocation.athlete_ids),
                    "season": [season] * len(allocation.athlete_ids),
                    "price": medians,
                },
                schema_overrides={"season": pl.Int64},
            )
        )
    allocation_medians = pl.concat(prices) if prices else None
    return fit_market_labels(
        table, deals, registry, allocation_medians=allocation_medians, matched=matched
    )


def value_player(
    name: str,
    season: int,
    *,
    team: str | None = None,
    sport: str = "mbb",
    labels: Path | None = None,
    cache: ArtifactCache | None = None,
    registry: AssumptionRegistry | None = None,
    economics: bool = True,
    seed: int = 0,
    last_available_season: int | None = None,
) -> PlayerValuation:
    """Value one player-season: impact, wins, program value, market price and surplus.

    The price comes from, in order: disclosed pay for this player-season, the fitted
    market model when it passes its gates, then the roster-budget allocation.
    """
    if sport != "mbb":
        raise ValueError(f"sport {sport!r} is not modelled yet; v0.3 covers men's basketball")
    cache = cache or ArtifactCache.default()
    registry = registry or default_registry()
    model = fit_season(
        season, cache=cache, registry=registry, last_available_season=last_available_season
    )
    econ = fit_economics(season, cache=cache, registry=registry) if economics else None
    deals = load_deals(labels)
    market_fit: MarketFit | None = None
    market_econ = econ
    note: str | None = None
    try:
        last_label_season = max(_label_seasons(deals, registry))
        if market_econ is None or last_label_season > season:
            # The model is trained on revenue features, so the player's own features need
            # the same economics even when program value is switched off.
            market_econ = fit_economics(
                max(last_label_season, season), cache=cache, registry=registry
            )
        market_fit = fit_market(
            deals,
            cache=cache,
            registry=registry,
            fitted={season: model},
            economics=market_econ,
        )
    except InsufficientLabelsError as error:
        note = str(error)
    return compose_player(
        model,
        econ,
        registry,
        name,
        team=team,
        deals=deals,
        market_fit=market_fit,
        market_note=note,
        market_economics=market_econ,
        seed=seed,
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
    "InsufficientLabelsError",
    "fit_economics",
    "fit_market",
    "fit_season",
    "load_deals",
    "load_economics_frames",
    "validate",
    "value_player",
    "value_team",
]
