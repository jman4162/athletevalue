"""Download and read men's basketball inputs, and the box-score prior's training data."""

from __future__ import annotations

import hashlib
import json

import polars as pl

from athletevalue.assumptions.registry import AssumptionRegistry
from athletevalue.frames.box import player_box_totals
from athletevalue.impact.prior import BoxPriorModel, fit_box_prior
from athletevalue.sources.cache import ArtifactCache
from athletevalue.sources.sportsdataverse import (
    FIRST_POSSESSION_SEASON,
    SdvClient,
    SdvDataset,
    SeasonUnavailableError,
)
from athletevalue.valuation.season import SeasonFrames, baseline_rapm
from athletevalue.versions import MODEL_VERSION

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
    """The *count* seasons nearest *target*, excluding it, earlier seasons first."""
    others = [s for s in available if s != target]
    ranked = sorted(others, key=lambda s: (abs(s - target), s > target))
    return sorted(ranked[:count])


def cached_baseline(
    season: int, cache: ArtifactCache, registry: AssumptionRegistry
) -> pl.DataFrame:
    """No-prior rating table for *season*, from the cache when the same settings made it."""
    settings = json.dumps(
        {key: registry.get(key).value for key in _BASELINE_KEYS}, sort_keys=True, default=str
    )
    digest = hashlib.sha256(f"{MODEL_VERSION}{settings}".encode()).hexdigest()[:12]
    relative = f"derived/rapm/baseline_{season}_{digest}.parquet"
    target = cache.path_for(relative)
    if target.exists():
        return pl.read_parquet(target)
    table = baseline_rapm(load_season_frames(season, cache), registry).table
    target.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(target)
    return table


def load_box_prior(
    target: int, cache: ArtifactCache, registry: AssumptionRegistry, *, last_season: int
) -> BoxPriorModel:
    """Fit the box-score prior for *target* on other seasons' no-prior ratings."""
    client = SdvClient(cache)
    available = list(range(FIRST_POSSESSION_SEASON, last_season + 1))
    count = int(registry.get("mbb.prior.training_seasons").scalar())
    training = []
    for season in training_seasons(target, count, available):
        try:
            box = player_box_totals(client.frame(SdvDataset.PLAYER_BOX, season))
            training.append((season, cached_baseline(season, cache, registry), box))
        except SeasonUnavailableError:
            continue
    return fit_box_prior(
        training,
        pseudo_possessions=registry.get("mbb.prior.rate_pseudo_possessions").scalar(),
        ridge=registry.get("mbb.prior.box_ridge").scalar(),
    )
