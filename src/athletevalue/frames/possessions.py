"""Possessions to lineup rows: the unit the RAPM design matrix is built from.

Each output row is every possession in one game with the same offense team, the
same five offensive players, the same five defensive players and the same venue.
Summing points and counting possessions within a row loses nothing a
possession-weighted ridge regression uses, and shrinks ~800k possessions to a
few hundred thousand rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from athletevalue.frames.columns import POSSESSION_COLUMNS, SLOTS, require_columns

NON_D1 = "non_d1"


@dataclass(frozen=True)
class LineupData:
    rows: pl.DataFrame
    """contest_id, offense_team, defense_team, offense_d1, defense_d1, venue,
    off_ids (list[str]), def_ids (list[str]), poss, pts."""

    players: pl.DataFrame
    """athlete_id, name, team, off_poss, def_poss."""

    counts: dict[str, int]
    """Possessions kept and dropped at each step, for the run report."""


def build_lineup_data(
    possessions: pl.DataFrame,
    *,
    d1_teams: frozenset[str],
    neutral_espn_games: frozenset[str],
    drop_garbage_time: bool,
) -> LineupData:
    require_columns(possessions, POSSESSION_COLUMNS, "possessions")
    counts = {"input": possessions.height}
    frame = possessions
    if drop_garbage_time:
        frame = frame.filter(pl.col("is_garbage_time").fill_null(0) == 0)
    counts["after_garbage_time"] = frame.height

    d1 = pl.Series(sorted(d1_teams), dtype=pl.Utf8).implode()
    frame = frame.with_columns(
        pl.col("home").is_in(d1).alias("home_d1"),
        pl.col("away").is_in(d1).alias("away_d1"),
        _ids_complete("home").alias("home_ids_ok"),
        _ids_complete("away").alias("away_ids_ok"),
    )
    usable = (
        (pl.col("home_d1") | pl.col("away_d1"))
        & (~pl.col("home_d1") | pl.col("home_ids_ok"))
        & (~pl.col("away_d1") | pl.col("away_ids_ok"))
    )
    frame = frame.filter(usable)
    counts["usable"] = frame.height

    neutral = pl.Series(sorted(neutral_espn_games), dtype=pl.Utf8).implode()
    offense_home = pl.col("poss_team") == pl.col("home")
    frame = frame.with_columns(
        offense_home.alias("offense_home"),
        pl.col("espn_game_id").cast(pl.Utf8).is_in(neutral).alias("neutral"),
        _sorted_ids("home").alias("home_ids"),
        _sorted_ids("away").alias("away_ids"),
    )
    rows = (
        frame.with_columns(
            pl.when(offense_home)
            .then(pl.col("home"))
            .otherwise(pl.col("away"))
            .alias("offense_team"),
            pl.when(offense_home)
            .then(pl.col("away"))
            .otherwise(pl.col("home"))
            .alias("defense_team"),
            pl.when(offense_home)
            .then(pl.col("home_d1"))
            .otherwise(pl.col("away_d1"))
            .alias("offense_d1"),
            pl.when(offense_home)
            .then(pl.col("away_d1"))
            .otherwise(pl.col("home_d1"))
            .alias("defense_d1"),
            pl.when(pl.col("neutral"))
            .then(0)
            .when(offense_home)
            .then(1)
            .otherwise(-1)
            .cast(pl.Int8)
            .alias("venue"),
            pl.when(offense_home)
            .then(pl.col("home_ids"))
            .otherwise(pl.col("away_ids"))
            .alias("off_key"),
            pl.when(offense_home)
            .then(pl.col("away_ids"))
            .otherwise(pl.col("home_ids"))
            .alias("def_key"),
        )
        .group_by(
            "contest_id",
            "offense_team",
            "defense_team",
            "offense_d1",
            "defense_d1",
            "venue",
            "off_key",
            "def_key",
            maintain_order=True,
        )
        .agg(pl.len().alias("poss"), pl.col("pts").sum().alias("pts"))
        .with_columns(
            _split_key("off_key").alias("off_ids"),
            _split_key("def_key").alias("def_ids"),
        )
        .drop("off_key", "def_key")
    )
    counts["lineup_rows"] = rows.height
    return LineupData(rows=rows, players=_player_table(frame), counts=counts)


def _ids_complete(side: str) -> pl.Expr:
    return pl.all_horizontal([pl.col(f"{side}_{slot}_player_id").is_not_null() for slot in SLOTS])


def _sorted_ids(side: str) -> pl.Expr:
    ids = pl.concat_list([pl.col(f"{side}_{slot}_player_id").cast(pl.Utf8) for slot in SLOTS])
    return (
        pl.when(pl.col(f"{side}_d1")).then(ids.list.sort().list.join("|")).otherwise(pl.lit(NON_D1))
    )


def _split_key(column: str) -> pl.Expr:
    return (
        pl.when(pl.col(column) == NON_D1)
        .then(pl.lit([], dtype=pl.List(pl.Utf8)))
        .otherwise(pl.col(column).str.split("|"))
    )


def _player_table(frame: pl.DataFrame) -> pl.DataFrame:
    parts: list[pl.DataFrame] = []
    for side in ("home", "away"):
        is_offense = pl.col("offense_home") if side == "home" else ~pl.col("offense_home")
        for slot in SLOTS:
            parts.append(
                frame.filter(pl.col(f"{side}_d1")).select(
                    pl.col(f"{side}_{slot}_player_id").cast(pl.Utf8).alias("athlete_id"),
                    pl.col(f"{side}_{slot}_clean_name").cast(pl.Utf8).alias("name"),
                    pl.col(side).alias("team"),
                    is_offense.alias("on_offense"),
                )
            )
    long = pl.concat(parts)
    per_team = long.group_by("athlete_id", "team").agg(
        pl.col("name").drop_nulls().mode().first().alias("name"),
        pl.col("on_offense").sum().cast(pl.Int64).alias("off_poss"),
        (~pl.col("on_offense")).sum().cast(pl.Int64).alias("def_poss"),
    )
    # A player who changed teams mid-season is listed under the team where he played most.
    return (
        per_team.with_columns((pl.col("off_poss") + pl.col("def_poss")).alias("_total"))
        .sort(["athlete_id", "_total"], descending=[False, True])
        .group_by("athlete_id", maintain_order=True)
        .agg(
            pl.col("name").first(),
            pl.col("team").first(),
            pl.col("off_poss").sum(),
            pl.col("def_poss").sum(),
        )
        .sort("athlete_id")
    )
