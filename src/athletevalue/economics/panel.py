"""School-season panel joining EADA men's basketball finances to results."""

from __future__ import annotations

import polars as pl

EADA_SPORT = "Basketball"


def revenue_panel(
    outcomes: pl.DataFrame, eada: pl.DataFrame, crosswalk: pl.DataFrame
) -> pl.DataFrame:
    """unitid, team, conference, season, rev_men, exp_men, allocated, games, wins,
    ncaa_bid, wins_lag, bid_lag, has_lag.

    ``outcomes`` has one row per team-season (team, season, conference, games, wins,
    ncaa_bid). ``eada`` has EADA sport rows (unitid, season, Sports, REV_MEN, EXP_MEN).
    ``allocated`` marks rows where reported revenue equals expense, which is how many
    schools book a sport whose revenue they do not track separately.
    """
    finances = (
        eada.filter(pl.col("Sports") == EADA_SPORT)
        .select(
            pl.col("unitid").cast(pl.Int64),
            pl.col("season").cast(pl.Int64),
            pl.col("REV_MEN").cast(pl.Float64).alias("rev_men"),
            pl.col("EXP_MEN").cast(pl.Float64).alias("exp_men"),
        )
        .unique(subset=["unitid", "season"], keep="first")
    )
    keyed = outcomes.select(
        "team",
        pl.col("season").cast(pl.Int64),
        "conference",
        pl.col("games").cast(pl.Int64),
        pl.col("wins").cast(pl.Int64),
        pl.col("ncaa_bid").cast(pl.Boolean),
    ).join(
        crosswalk.drop_nulls("unitid").select("team", pl.col("unitid").cast(pl.Int64)),
        on="team",
        how="inner",
    )
    return (
        keyed.join(finances, on=["unitid", "season"], how="inner")
        .unique(subset=["team", "season"], keep="first")
        .sort("team", "season")
        .with_columns(
            (pl.col("rev_men") == pl.col("exp_men")).alias("allocated"),
            pl.col("wins").shift(1).over("team").alias("wins_lag"),
            pl.col("ncaa_bid").shift(1).over("team").alias("bid_lag"),
            (pl.col("season").shift(1).over("team") == pl.col("season") - 1).alias("has_lag"),
        )
        .with_columns(pl.col("has_lag").fill_null(False))
    )


def revenue_base(
    panel: pl.DataFrame, team: str, *, through_season: int, n_seasons: int
) -> float | None:
    """Mean reported men's basketball revenue over a team's latest seasons.

    Allocated rows are kept here: they misstate how revenue responds to winning, but
    they still give the level of spending the program books against the sport.
    """
    recent = (
        panel.filter(
            (pl.col("team") == team)
            & (pl.col("season") <= through_season)
            & (pl.col("rev_men") > 0)
        )
        .sort("season", descending=True)
        .head(n_seasons)
    )
    if recent.is_empty():
        return None
    return float(recent["rev_men"].mean())  # type: ignore[arg-type]
