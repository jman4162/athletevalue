"""Season box-score totals per player, from the SportsDataverse player box file."""

from __future__ import annotations

import polars as pl

from athletevalue.frames.columns import require_columns

BOX_STATS: tuple[str, ...] = (
    "pts",
    "fga",
    "tpa",
    "fta",
    "rima",
    "mida",
    "ast",
    "tov",
    "orb",
    "drb",
    "stl",
    "blk",
    "pf",
)
"""Counting stats used by the box-score prior. ``rima`` and ``mida`` are rim and
mid-range attempts; ``tpa`` is three-point attempts."""

PLAYER_BOX_COLUMNS: tuple[str, ...] = ("player_id", "o_poss", *BOX_STATS)


def player_box_totals(box: pl.DataFrame) -> pl.DataFrame:
    """athlete_id, o_poss and each stat in ``BOX_STATS``, summed over the season."""
    require_columns(box, PLAYER_BOX_COLUMNS, "player box")
    return (
        box.filter(pl.col("player_id").is_not_null())
        .group_by(pl.col("player_id").cast(pl.Utf8).alias("athlete_id"))
        .agg(
            pl.col("o_poss").cast(pl.Float64).sum(),
            *(pl.col(stat).cast(pl.Float64).fill_null(0).sum() for stat in BOX_STATS),
        )
        .sort("athlete_id")
    )
