"""Download and read men's basketball inputs, and the box-score prior's training data."""

from __future__ import annotations

import polars as pl

from athletevalue.assumptions.registry import AssumptionRegistry
from athletevalue.frames.box import player_box_totals
from athletevalue.impact.prior import BoxPriorModel, PriorTrainingError, fit_box_prior
from athletevalue.sources.cache import ArtifactCache, derived_path
from athletevalue.sources.sportsdataverse import (
    FIRST_POSSESSION_SEASON,
    SdvClient,
    SdvDataset,
    SeasonUnavailableError,
)
from athletevalue.valuation.season import SeasonFrames, baseline_rapm

_INPUTS = (
    SdvDataset.POSSESSIONS,
    SdvDataset.TEAM_IDS,
    SdvDataset.SCHEDULE,
    SdvDataset.ESPN_SCHEDULE,
    SdvDataset.PLAYER_BOX,
)

_BASELINE_KEYS = (
    "mbb.impact.ridge_lambda",
    "mbb.impact.min_possessions",
    "mbb.impact.drop_garbage_time",
)


def load_season_frames(season: int, cache: ArtifactCache) -> SeasonFrames:
    client = SdvClient(cache)
    return SeasonFrames(
        season=season,
        possessions=client.frame(SdvDataset.POSSESSIONS, season),
        team_ids=client.frame(SdvDataset.TEAM_IDS, season),
        schedule=client.frame(SdvDataset.SCHEDULE, season),
        espn_schedule=client.frame(SdvDataset.ESPN_SCHEDULE, season),
        player_box=client.frame(SdvDataset.PLAYER_BOX, season),
        sources=tuple(client.reference(dataset, season) for dataset in _INPUTS),
    )


def training_seasons(target: int, count: int, available: list[int]) -> list[int]:
    """The *count* most recent seasons before *target*.

    Later seasons are never used: a prior fitted on them would carry information
    from the future into a season's ratings, and any backtest built on those
    ratings would overstate what could have been known at the time.
    """
    earlier = sorted(s for s in available if s < target)
    return earlier[-count:] if count > 0 else []


def cached_baseline(
    season: int, cache: ArtifactCache, registry: AssumptionRegistry
) -> pl.DataFrame:
    """No-prior rating table for *season*, from the cache when the same settings made it."""
    settings = {key: registry.get(key).value for key in _BASELINE_KEYS}
    relative = derived_path("rapm", f"baseline_{season}", settings)
    cached = cache.read_derived(relative)
    if cached is not None:
        return cached
    table = baseline_rapm(load_season_frames(season, cache), registry).table
    client = SdvClient(cache)
    sources = [client.artifact(dataset, season) for dataset in _INPUTS]
    cache.write_derived(relative, table, sources=sources, settings=settings)
    return table


def prior_training_candidates(
    target: int, registry: AssumptionRegistry, *, last_season: int
) -> list[int]:
    """Seasons the box-score prior for *target* would train on, before availability checks."""
    available = list(range(FIRST_POSSESSION_SEASON, min(last_season, target - 1) + 1))
    count = int(registry.get("mbb.prior.training_seasons").scalar())
    return training_seasons(target, count, available)


def fetch_prior_inputs(
    target: int, cache: ArtifactCache, registry: AssumptionRegistry, *, last_season: int
) -> list[int]:
    """Download the files the prior for *target* trains on. Returns the seasons found."""
    client = SdvClient(cache)
    found = []
    for season in prior_training_candidates(target, registry, last_season=last_season):
        try:
            for dataset in _INPUTS:
                client.artifact(dataset, season)
        except SeasonUnavailableError:
            continue
        found.append(season)
    return found


def load_box_prior(
    target: int, cache: ArtifactCache, registry: AssumptionRegistry, *, last_season: int
) -> BoxPriorModel:
    """Fit the box-score prior for *target* on earlier seasons' no-prior ratings.

    *last_season* caps the seasons considered; seasons at or after *target* are
    excluded regardless.
    """
    client = SdvClient(cache)
    training = []
    for season in prior_training_candidates(target, registry, last_season=last_season):
        try:
            box = player_box_totals(client.frame(SdvDataset.PLAYER_BOX, season))
            training.append((season, cached_baseline(season, cache, registry), box))
        except SeasonUnavailableError:
            continue
    if not training:
        raise PriorTrainingError(f"no season before {target} has possession data")
    return fit_box_prior(
        training,
        pseudo_possessions=registry.get("mbb.prior.rate_pseudo_possessions").scalar(),
        ridge=registry.get("mbb.prior.box_ridge").scalar(),
    )
