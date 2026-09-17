"""Tournament games recovered by name matching in seasons without ESPN ids.

Possession files carry ESPN game ids from 2023 on. Earlier seasons are matched by
date and names, and a missed NCAA tournament game drops a bid or a tournament win
from the economics inputs. Every tournament game in the ESPN schedule is matched
except where the stats.ncaa.org schedule has no such game (one in 2011).
"""

from __future__ import annotations

import polars as pl
import pytest

from athletevalue.frames.columns import NCAA_TOURNAMENT_ID, POSTSEASON_TYPE
from athletevalue.frames.games import combined_game_map
from athletevalue.sources.cache import ArtifactCache
from athletevalue.sources.sportsdataverse import SdvClient, SdvDataset

pytestmark = pytest.mark.network

MISSING_FROM_SOURCE = {2011: 1}


@pytest.mark.parametrize("season", [2011, 2014, 2016, 2017, 2019, 2021, 2022])
def test_pre_2023_tournament_and_neutral_games_are_matched(season):
    client = SdvClient(ArtifactCache.default())
    possessions = (
        pl.scan_parquet(client.artifact(SdvDataset.POSSESSIONS, season).path)
        .select("contest_id", "espn_game_id")
        .collect()
    )
    schedule = client.frame(SdvDataset.SCHEDULE, season)
    espn = client.frame(SdvDataset.ESPN_SCHEDULE, season)
    game_map = combined_game_map(possessions, schedule, espn)
    ids = game_map["espn_game_id"].implode()

    tournament = espn.filter(
        (pl.col("season_type") == POSTSEASON_TYPE) & (pl.col("tournament_id") == NCAA_TOURNAMENT_ID)
    )
    matched = tournament.filter(pl.col("game_id").cast(pl.Utf8).is_in(ids)).height
    assert matched == tournament.height - MISSING_FROM_SOURCE.get(season, 0)

    neutral = espn.filter(
        pl.col("neutral_site").fill_null(False) | (pl.col("season_type") == POSTSEASON_TYPE)
    )
    share = neutral.filter(pl.col("game_id").cast(pl.Utf8).is_in(ids)).height / neutral.height
    assert share >= 0.99
    assert not game_map["contest_id"].is_duplicated().any()
    assert not game_map["espn_game_id"].is_duplicated().any()
