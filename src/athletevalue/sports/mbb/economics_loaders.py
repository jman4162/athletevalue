"""Download the multi-season inputs for the program-economics models."""

from __future__ import annotations

import polars as pl

from athletevalue.frames.games import build_season_games, combined_game_map
from athletevalue.identity.crosswalk import team_crosswalk
from athletevalue.sources.cache import ArtifactCache, RawArtifact, derived_path
from athletevalue.sources.eada import EadaClient, EadaYearUnavailableError
from athletevalue.sources.sportsdataverse import (
    FIRST_POSSESSION_SEASON,
    SdvClient,
    SdvDataset,
    SeasonUnavailableError,
)
from athletevalue.valuation.economy import EconomicsFrames

_OUTCOME_INPUTS = (
    SdvDataset.POSSESSIONS,
    SdvDataset.SCHEDULE,
    SdvDataset.ESPN_SCHEDULE,
    SdvDataset.TEAM_IDS,
)


def season_outcomes(season: int, client: SdvClient) -> tuple[pl.DataFrame, list[RawArtifact]]:
    """One season's team results, and the artifacts they were built from."""
    artifacts = {dataset: client.artifact(dataset, season) for dataset in _OUTCOME_INPUTS}
    possessions = (
        pl.scan_parquet(artifacts[SdvDataset.POSSESSIONS].path)
        .select("contest_id", "espn_game_id")
        .collect()
    )
    schedule = pl.read_parquet(artifacts[SdvDataset.SCHEDULE].path)
    espn = pl.read_parquet(artifacts[SdvDataset.ESPN_SCHEDULE].path)
    games = build_season_games(
        schedule,
        espn,
        pl.read_parquet(artifacts[SdvDataset.TEAM_IDS].path),
        combined_game_map(possessions, schedule, espn),
    )
    frame = games.teams.with_columns(pl.lit(season, dtype=pl.Int64).alias("season"))
    return frame, list(artifacts.values())


def load_economics_frames(cache: ArtifactCache, *, last_season: int) -> EconomicsFrames:
    """Results for every season with possession data through *last_season*, and EADA finances."""
    seasons = range(FIRST_POSSESSION_SEASON, last_season + 1)
    derived = derived_path("economics", f"outcomes_{FIRST_POSSESSION_SEASON}_{last_season}")
    outcomes = cache.read_derived(derived)
    if outcomes is None:
        sdv = SdvClient(cache)
        parts, artifacts = [], []
        for season in seasons:
            try:
                frame, used = season_outcomes(season, sdv)
            except SeasonUnavailableError:
                continue
            parts.append(frame)
            artifacts.extend(used)
        outcomes = pl.concat(parts, how="vertical_relaxed")
        cache.write_derived(derived, outcomes, sources=artifacts)
    present = set(outcomes["season"].unique().to_list())
    notes = [
        f"no SportsDataverse results for {season}; the economics fit skips it"
        for season in seasons
        if season not in present
    ]

    eada_client = EadaClient(cache)
    eada_parts = []
    sources = []
    for season in seasons:
        try:
            eada_parts.append(eada_client.schools(season))
            sources.append(eada_client.reference(season))
        except EadaYearUnavailableError as error:
            notes.append(f"{error}; the revenue panel skips it")
    return EconomicsFrames(
        outcomes=outcomes,
        eada=pl.concat(eada_parts, how="vertical_relaxed"),
        crosswalk=team_crosswalk(),
        sources=tuple(sources),
        notes=tuple(notes),
    )
