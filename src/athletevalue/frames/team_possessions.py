"""Team offensive and defensive possessions and points, over all possessions."""

from __future__ import annotations

import polars as pl

from athletevalue.constants import PER_100


def team_possession_totals(possessions: pl.DataFrame) -> pl.DataFrame:
    """team, games_with_pbp, off_poss, def_poss, pts_for, pts_against, ortg, drtg, pace.

    Garbage time is included: these totals describe how a team's season actually
    went, which is what win percentage responds to.
    """
    base = possessions.select(
        "contest_id",
        "home",
        "away",
        "poss_team",
        pl.col("pts").cast(pl.Int64),
    ).with_columns(
        pl.when(pl.col("poss_team") == pl.col("home"))
        .then(pl.col("away"))
        .otherwise(pl.col("home"))
        .alias("defense_team")
    )
    offense = base.group_by(pl.col("poss_team").alias("team")).agg(
        pl.len().alias("off_poss"),
        pl.col("pts").sum().alias("pts_for"),
        pl.col("contest_id").n_unique().alias("games_with_pbp"),
    )
    defense = base.group_by(pl.col("defense_team").alias("team")).agg(
        pl.len().alias("def_poss"), pl.col("pts").sum().alias("pts_against")
    )
    return (
        offense.join(defense, on="team", how="inner")
        .with_columns(
            (PER_100 * pl.col("pts_for") / pl.col("off_poss")).alias("ortg"),
            (PER_100 * pl.col("pts_against") / pl.col("def_poss")).alias("drtg"),
            ((pl.col("off_poss") + pl.col("def_poss")) / 2 / pl.col("games_with_pbp")).alias(
                "pace"
            ),
        )
        .sort("team")
    )
