"""Game results, venues and NCAA tournament participation for one season."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from athletevalue.frames.columns import (
    ESPN_SCHEDULE_COLUMNS,
    MATCH_DAY_WINDOW,
    MATCH_MIN_SIMILARITY,
    NCAA_TOURNAMENT_ID,
    POSTSEASON_TYPE,
    SCHEDULE_COLUMNS,
    TEAM_ID_COLUMNS,
    require_columns,
)
from athletevalue.identity.names import similarity


@dataclass(frozen=True)
class SeasonGames:
    games: pl.DataFrame
    """contest_id, home, away, home_score, away_score, neutral, ncaa_tournament."""

    teams: pl.DataFrame
    """team, conference, games, wins, losses, points_for, points_against,
    ncaa_bid, ncaa_games, ncaa_wins."""


def espn_game_map(possessions: pl.DataFrame) -> pl.DataFrame:
    """contest_id to ESPN game id, where the possession file records it (2023 onward)."""
    return (
        possessions.select("contest_id", pl.col("espn_game_id").cast(pl.Utf8))
        .drop_nulls("espn_game_id")
        .unique(subset=["contest_id"], keep="first")
    )


def match_espn_games(schedule: pl.DataFrame, espn_schedule: pl.DataFrame) -> pl.DataFrame:
    """contest_id to ESPN game id for neutral-site and postseason games, by date and names.

    Only those games change a venue or a tournament flag, so only they are matched.
    A pair must fall within a day of each other and both team names must be similar;
    the best-scoring pairs are taken one to one.
    """
    empty = pl.DataFrame(schema={"contest_id": pl.Utf8, "espn_game_id": pl.Utf8})
    ncaa = schedule.select(
        "contest_id",
        pl.col("game_date").str.strptime(pl.Date, "%m/%d/%Y", strict=False).alias("date"),
        "home",
        "away",
    ).drop_nulls("date")
    espn = (
        espn_schedule.filter(
            pl.col("neutral_site").fill_null(False) | (pl.col("season_type") == POSTSEASON_TYPE)
        )
        .select(
            pl.col("game_id").cast(pl.Utf8).alias("espn_game_id"),
            pl.col("game_date").cast(pl.Date).alias("espn_date"),
            pl.col("home_location").alias("espn_home"),
            pl.col("away_location").alias("espn_away"),
        )
        .drop_nulls("espn_date")
    )
    if espn.is_empty() or ncaa.is_empty():
        return empty
    shifted = pl.concat(
        [
            espn.with_columns((pl.col("espn_date") + pl.duration(days=k)).alias("date"))
            for k in range(-MATCH_DAY_WINDOW, MATCH_DAY_WINDOW + 1)
        ]
    )
    pairs = shifted.join(ncaa, on="date", how="inner")
    if pairs.is_empty():
        return empty
    cache: dict[tuple[str, str], float] = {}

    def sim(left: str | None, right: str | None) -> float:
        if left is None or right is None:
            return 0.0
        key = (left, right)
        if key not in cache:
            cache[key] = similarity(left, right)
        return cache[key]

    scores = [
        max(sim(h, eh) + sim(a, ea), sim(h, ea) + sim(a, eh)) / 2
        for h, a, eh, ea in zip(
            pairs["home"], pairs["away"], pairs["espn_home"], pairs["espn_away"], strict=True
        )
    ]
    ranked = (
        pairs.with_columns(pl.Series("score", scores, dtype=pl.Float64))
        .filter(pl.col("score") >= MATCH_MIN_SIMILARITY)
        .sort("score", descending=True)
    )
    taken_contests: set[str] = set()
    taken_espn: set[str] = set()
    matched: list[tuple[str, str]] = []
    for contest, espn_id in zip(ranked["contest_id"], ranked["espn_game_id"], strict=True):
        if contest in taken_contests or espn_id in taken_espn:
            continue
        taken_contests.add(contest)
        taken_espn.add(espn_id)
        matched.append((contest, espn_id))
    if not matched:
        return empty
    return pl.DataFrame(
        {"contest_id": [c for c, _ in matched], "espn_game_id": [e for _, e in matched]},
        schema={"contest_id": pl.Utf8, "espn_game_id": pl.Utf8},
    )


def combined_game_map(
    possessions: pl.DataFrame, schedule: pl.DataFrame, espn_schedule: pl.DataFrame
) -> pl.DataFrame:
    """Possession-file ESPN ids where present, date-and-name matches elsewhere."""
    direct = espn_game_map(possessions)
    missing = schedule.join(direct, on="contest_id", how="anti")
    if missing.is_empty():
        return direct
    used = direct["espn_game_id"].implode()
    remaining = espn_schedule.filter(~pl.col("game_id").cast(pl.Utf8).is_in(used))
    return pl.concat([direct, match_espn_games(missing, remaining)])


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
