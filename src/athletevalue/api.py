"""Public entry points that load data and run the models."""

from __future__ import annotations

from athletevalue.assumptions.registry import AssumptionRegistry, default_registry
from athletevalue.sources.cache import ArtifactCache
from athletevalue.sources.sportsdataverse import SdvClient, SdvDataset
from athletevalue.sources.torvik import TorvikClient
from athletevalue.sports.mbb.loaders import load_season_frames
from athletevalue.valuation.season import SeasonModel, assemble_season
from athletevalue.valuation.validate import ValidationReport, validate_season


def fit_season(
    season: int,
    *,
    cache: ArtifactCache | None = None,
    registry: AssumptionRegistry | None = None,
    lam: float | None = None,
    choose_lambda_by_cv: bool = False,
) -> SeasonModel:
    """Download (or reuse) one season's inputs and fit RAPM, team ratings and win models."""
    cache = cache or ArtifactCache.default()
    registry = registry or default_registry()
    frames = load_season_frames(season, cache)
    return assemble_season(frames, registry, lam=lam, choose_lambda_by_cv=choose_lambda_by_cv)


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
