"""Download the multi-season inputs for the program-economics models."""

from __future__ import annotations

import polars as pl

from athletevalue.frames.games import build_season_games, combined_game_map
from athletevalue.identity.crosswalk import team_crosswalk
from athletevalue.sources.cache import ArtifactCache
from athletevalue.sources.eada import EadaClient, EadaYearUnavailableError
from athletevalue.sources.sportsdataverse import (
    FIRST_POSSESSION_SEASON,
    SdvClient,
    SdvDataset,
    SeasonUnavailableError,
)
from athletevalue.valuation.economy import EconomicsFrames


def season_outcomes(season: int, client: SdvClient) -> pl.DataFrame:
    possessions = (
        pl.scan_parquet(client.artifact(SdvDataset.POSSESSIONS, season).path)
        .select("contest_id", "espn_game_id")
        .collect()
    )
    schedule = client.frame(SdvDataset.SCHEDULE, season)
    espn = client.frame(SdvDataset.ESPN_SCHEDULE, season)
    games = build_season_games(
        schedule,
        espn,
        client.frame(SdvDataset.TEAM_IDS, season),
        combined_game_map(possessions, schedule, espn),
    )
    return games.teams.with_columns(pl.lit(season, dtype=pl.Int64).alias("season"))


def load_economics_frames(cache: ArtifactCache, *, last_season: int) -> EconomicsFrames:
    """Results for every season with possession data through *last_season*, and EADA finances."""
    derived = f"derived/economics/outcomes_{FIRST_POSSESSION_SEASON}_{last_season}.parquet"
    target = cache.path_for(derived)
    sdv = SdvClient(cache)
    if target.exists():
        outcomes = pl.read_parquet(target)
    else:
        parts = []
        for season in range(FIRST_POSSESSION_SEASON, last_season + 1):
            try:
                parts.append(season_outcomes(season, sdv))
            except SeasonUnavailableError:
                continue
        outcomes = pl.concat(parts, how="vertical_relaxed")
        target.parent.mkdir(parents=True, exist_ok=True)
        outcomes.write_parquet(target)

    eada_client = EadaClient(cache)
    eada_parts = []
    sources = []
    for season in range(FIRST_POSSESSION_SEASON, last_season + 1):
        try:
            eada_parts.append(eada_client.schools(season))
            sources.append(eada_client.reference(season))
        except EadaYearUnavailableError:
            continue
    return EconomicsFrames(
        outcomes=outcomes,
        eada=pl.concat(eada_parts, how="vertical_relaxed"),
        crosswalk=team_crosswalk(),
        sources=tuple(sources),
    )
