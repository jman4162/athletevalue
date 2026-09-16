"""Game results, venues and NCAA tournament participation for one season."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from athletevalue.frames.columns import (
    ESPN_SCHEDULE_COLUMNS,
    NCAA_TOURNAMENT_ID,
    POSTSEASON_TYPE,
    SCHEDULE_COLUMNS,
    TEAM_ID_COLUMNS,
    require_columns,
)


@dataclass(frozen=True)
class SeasonGames:
    games: pl.DataFrame
    """contest_id, home, away, home_score, away_score, neutral, ncaa_tournament."""

    teams: pl.DataFrame
    """team, conference, games, wins, losses, points_for, points_against,
    ncaa_bid, ncaa_games, ncaa_wins."""


def espn_game_map(possessions: pl.DataFrame) -> pl.DataFrame:
    """contest_id to ESPN game id and ESPN team ids, taken from the possession file."""
    return possessions.select(
        "contest_id",
        pl.col("espn_game_id").cast(pl.Utf8),
        "home",
        "away",
        pl.col("home_espn_team_id").cast(pl.Utf8),
        pl.col("away_espn_team_id").cast(pl.Utf8),
    ).unique(subset=["contest_id"], keep="first")


def build_season_games(
    schedule: pl.DataFrame,
    espn_schedule: pl.DataFrame,
    team_ids: pl.DataFrame,
    game_map: pl.DataFrame,
) -> SeasonGames:
    require_columns(schedule, SCHEDULE_COLUMNS, "schedule")
    require_columns(espn_schedule, ESPN_SCHEDULE_COLUMNS, "ESPN schedule")
    require_columns(team_ids, TEAM_ID_COLUMNS, "team ids")

    espn = espn_schedule.select(
        pl.col("game_id").cast(pl.Utf8).alias("espn_game_id"),
        pl.col("neutral_site").fill_null(False).alias("neutral"),
        (
            (pl.col("season_type") == POSTSEASON_TYPE)
            & (pl.col("tournament_id") == NCAA_TOURNAMENT_ID)
        )
        .fill_null(False)
        .alias("ncaa_tournament"),
    )
    games = (
        schedule.select(*SCHEDULE_COLUMNS)
        .filter(pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null())
        .join(game_map.select("contest_id", "espn_game_id"), on="contest_id", how="left")
        .join(espn, on="espn_game_id", how="left")
        .with_columns(
            pl.col("neutral").fill_null(False), pl.col("ncaa_tournament").fill_null(False)
        )
    )

    long = pl.concat(
        [
            games.select(
                pl.col(team).alias("team"),
                pl.col(f"{team}_score").alias("points_for"),
                pl.col(f"{other}_score").alias("points_against"),
                "ncaa_tournament",
            )
            for team, other in (("home", "away"), ("away", "home"))
        ]
    ).with_columns((pl.col("points_for") > pl.col("points_against")).alias("won"))

    records = long.group_by("team").agg(
        pl.len().alias("games"),
        pl.col("won").sum().cast(pl.Int64).alias("wins"),
        pl.col("points_for").sum().alias("points_for"),
        pl.col("points_against").sum().alias("points_against"),
        pl.col("ncaa_tournament").any().alias("ncaa_bid"),
        pl.col("ncaa_tournament").sum().cast(pl.Int64).alias("ncaa_games"),
        (pl.col("ncaa_tournament") & pl.col("won")).sum().cast(pl.Int64).alias("ncaa_wins"),
    )
    teams = (
        team_ids.select("team", "conference")
        .unique(subset=["team"])
        .join(records, on="team", how="left")
        .with_columns(
            pl.col("games").fill_null(0),
            pl.col("wins").fill_null(0),
            pl.col("ncaa_bid").fill_null(False),
            pl.col("ncaa_games").fill_null(0),
            pl.col("ncaa_wins").fill_null(0),
        )
        .with_columns((pl.col("games") - pl.col("wins")).alias("losses"))
        .sort("team")
    )
    return SeasonGames(games=games, teams=teams)
