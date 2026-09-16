"""Team ratings implied by the fitted player coefficients."""

from __future__ import annotations

import numpy as np
import polars as pl

from athletevalue.frames.possessions import LineupData
from athletevalue.impact.design import Design
from athletevalue.impact.ridge import RidgeFit


def team_ratings(lineups: LineupData, design: Design, fit: RidgeFit) -> pl.DataFrame:
    """team, adj_off, adj_def, adj_net, lineup_off_poss, lineup_def_poss.

    ``adj_off`` is the possession-weighted average of the offensive coefficients on
    the floor for the team, relative to an average D1 lineup; ``adj_def`` likewise for
    defense, oriented so larger is better. Venue and opponents are already removed.
    """
    beta = fit.beta
    off_cols = design.offense_columns()
    def_cols = design.defense_columns()
    off_sum = np.asarray(design.X[:, off_cols] @ beta[off_cols]).ravel()
    def_sum = -np.asarray(design.X[:, def_cols] @ beta[def_cols]).ravel()
    rows = lineups.rows.select(
        "offense_team", "defense_team", "offense_d1", "defense_d1"
    ).with_columns(
        pl.Series("w", design.w), pl.Series("off_sum", off_sum), pl.Series("def_sum", def_sum)
    )
    offense = (
        rows.filter(pl.col("offense_d1"))
        .group_by(pl.col("offense_team").alias("team"))
        .agg(
            ((pl.col("w") * pl.col("off_sum")).sum() / pl.col("w").sum()).alias("adj_off"),
            pl.col("w").sum().alias("lineup_off_poss"),
        )
    )
    defense = (
        rows.filter(pl.col("defense_d1"))
        .group_by(pl.col("defense_team").alias("team"))
        .agg(
            ((pl.col("w") * pl.col("def_sum")).sum() / pl.col("w").sum()).alias("adj_def"),
            pl.col("w").sum().alias("lineup_def_poss"),
        )
    )
    return (
        offense.join(defense, on="team", how="inner")
        .with_columns((pl.col("adj_off") + pl.col("adj_def")).alias("adj_net"))
        .sort("adj_net", descending=True)
    )
