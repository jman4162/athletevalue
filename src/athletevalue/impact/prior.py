"""A box-score prior for RAPM.

Ridge RAPM shrinks every player toward zero, an average D1 player. With one
season of college possessions that shrinkage dominates, so a starter's rating
moves only a few points from zero. A better prior mean is what the player's box
score predicts.

The prior is a weighted ridge regression of no-prior RAPM coefficients (offense
and defense separately) on per-100-possession box rates, fitted on *other*
seasons so it never sees the season it is applied to. Rates are stabilized by
adding pseudo-possessions at the league-average rate, which pulls a walk-on's
3-for-3 toward the mean. The fitted means enter the RAPM solve as the prior
offset ``b0`` in ``lambda * ||beta - b0||^2``.

This follows the structure of EvanMiya's Bayesian Performance Rating, which uses
a college-specific box model as the RAPM prior mean.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import NDArray

from athletevalue.constants import PER_100
from athletevalue.frames.box import BOX_STATS
from athletevalue.frames.possessions import LineupData
from athletevalue.impact.design import Design

FloatArray = NDArray[np.float64]


class PriorTrainingError(ValueError):
    pass


@dataclass(frozen=True)
class BoxPriorModel:
    features: tuple[str, ...]
    league_rates: FloatArray
    """League rate per possession for each stat, from the training seasons."""
    mean: FloatArray
    scale: FloatArray
    coef_off: FloatArray
    """Intercept followed by one coefficient per standardized feature."""
    coef_def: FloatArray
    pseudo_possessions: float
    train_seasons: tuple[int, ...]
    n_players: int
    r2_off: float
    r2_def: float
    """Possession-weighted in-sample R^2 against the training RAPM."""

    def design_matrix(self, totals: pl.DataFrame) -> FloatArray:
        rates = box_rates(totals, self.league_rates, self.pseudo_possessions)
        standardized = (rates - self.mean) / self.scale
        matrix: FloatArray = np.column_stack([np.ones(len(standardized)), standardized])
        return matrix

    def predict(self, totals: pl.DataFrame) -> pl.DataFrame:
        """athlete_id, prior_off, prior_def for every player in *totals*."""
        matrix = self.design_matrix(totals)
        return pl.DataFrame(
            {
                "athlete_id": totals["athlete_id"],
                "prior_off": matrix @ self.coef_off,
                "prior_def": matrix @ self.coef_def,
            }
        )


def box_rates(
    totals: pl.DataFrame, league_rates: FloatArray, pseudo_possessions: float
) -> FloatArray:
    """Per-100-possession rates with ``pseudo_possessions`` of league-average play added."""
    poss = totals["o_poss"].to_numpy()
    counts = totals.select(BOX_STATS).to_numpy()
    rates: FloatArray = (
        PER_100
        * (counts + pseudo_possessions * league_rates[None, :])
        / (poss + pseudo_possessions)[:, None]
    )
    return rates


def fit_box_prior(
    training: list[tuple[int, pl.DataFrame, pl.DataFrame]],
    *,
    pseudo_possessions: float,
    ridge: float,
) -> BoxPriorModel:
    """Fit on ``(season, rapm_table, box_totals)`` triples.

    ``rapm_table`` must come from a fit without a prior, so the box model learns
    what the possessions alone say. Pooled players are excluded: their coefficient
    is shared, not their own.
    """
    frames = []
    for season, table, totals in training:
        joined = table.filter(~pl.col("pooled")).join(totals, on="athlete_id", how="inner")
        frames.append(joined.with_columns(pl.lit(season).alias("season")))
    if not frames:
        raise PriorTrainingError("no training seasons")
    data = pl.concat(frames, how="vertical_relaxed")
    if data.height <= len(BOX_STATS) + 1:
        raise PriorTrainingError(f"only {data.height} players to fit {len(BOX_STATS)} features")

    counts = data.select(BOX_STATS).to_numpy()
    league = counts.sum(axis=0) / data["o_poss"].sum()
    rates = box_rates(data, league, pseudo_possessions)
    mean = rates.mean(axis=0)
    scale = rates.std(axis=0)
    scale[scale == 0] = 1.0
    matrix = np.column_stack([np.ones(len(rates)), (rates - mean) / scale])

    penalty = np.full(matrix.shape[1], ridge)
    penalty[0] = 0.0
    coefs: dict[str, FloatArray] = {}
    r2: dict[str, float] = {}
    for target, weight_column in (("orapm", "off_poss"), ("drapm", "def_poss")):
        w = data[weight_column].cast(pl.Float64).to_numpy()
        y = data[target].to_numpy()
        gram = matrix.T @ (matrix * w[:, None]) + np.diag(penalty)
        coef = np.linalg.solve(gram, matrix.T @ (w * y))
        fitted = matrix @ coef
        centered = y - np.average(y, weights=w)
        r2[target] = float(1 - np.sum(w * (y - fitted) ** 2) / np.sum(w * centered**2))
        coefs[target] = coef
    return BoxPriorModel(
        features=BOX_STATS,
        league_rates=league,
        mean=mean,
        scale=scale,
        coef_off=coefs["orapm"],
        coef_def=coefs["drapm"],
        pseudo_possessions=pseudo_possessions,
        train_seasons=tuple(sorted({season for season, _, _ in training})),
        n_players=data.height,
        r2_off=r2["orapm"],
        r2_def=r2["drapm"],
    )


def prior_offset(design: Design, predictions: pl.DataFrame) -> FloatArray:
    """The ``b0`` vector for *design*: box predictions on player columns, zero elsewhere."""
    offset = np.zeros(design.n_columns)
    lookup = {
        row["athlete_id"]: (row["prior_off"], row["prior_def"])
        for row in predictions.iter_rows(named=True)
    }
    for index, athlete in enumerate(design.athlete_ids):
        if athlete in lookup:
            offset[design.off_col(index)], offset[design.def_col(index)] = lookup[athlete]
    return offset


def weighted_r2(actual: FloatArray, predicted: FloatArray, weights: FloatArray) -> float:
    centered = actual - np.average(actual, weights=weights)
    return float(1 - np.sum(weights * (actual - predicted) ** 2) / np.sum(weights * centered**2))


def team_adjust(
    predictions: pl.DataFrame,
    lineups: LineupData,
    design: Design,
    beta: FloatArray,
    rows: NDArray[np.bool_] | None = None,
) -> pl.DataFrame:
    """Shift each team's box predictions so they sum to the team's fitted rating.

    Box rates are not adjusted for opponents, so a low-major team's players look
    better in the box score than on the scoreboard. This is the Box Plus/Minus team
    adjustment: for each team and side, add the same amount to every player so the
    possession-weighted sum of priors equals the team's rating from *beta* (a
    no-prior fit). *rows* restricts both the team ratings and the possession
    weights to a subset of lineup rows, so cross-validation can exclude held-out
    games.
    """
    mask = np.ones(design.X.shape[0], dtype=bool) if rows is None else rows
    index = np.flatnonzero(mask)
    X = design.X[index]
    off_cols, def_cols = design.offense_columns(), design.defense_columns()
    frame = lineups.rows[index].with_columns(
        pl.Series("w", design.w[index]),
        pl.Series("off_sum", np.asarray(X[:, off_cols] @ beta[off_cols]).ravel()),
        pl.Series("def_sum", -np.asarray(X[:, def_cols] @ beta[def_cols]).ravel()),
    )
    sides = []
    for side, team_col, id_col, value_col, d1_col in (
        ("off", "offense_team", "off_ids", "off_sum", "offense_d1"),
        ("def", "defense_team", "def_ids", "def_sum", "defense_d1"),
    ):
        on_side = frame.filter(pl.col(d1_col))
        rating = on_side.group_by(pl.col(team_col).alias("team")).agg(
            ((pl.col("w") * pl.col(value_col)).sum() / pl.col("w").sum()).alias("rating"),
            pl.col("w").sum().alias("team_poss"),
        )
        shares = (
            on_side.select(pl.col(team_col).alias("team"), pl.col(id_col).alias("athlete_id"), "w")
            .explode("athlete_id", empty_as_null=False)
            .group_by("team", "athlete_id")
            .agg(pl.col("w").sum().alias("poss"))
            .join(rating, on="team")
            .with_columns((pl.col("poss") / pl.col("team_poss")).alias("share"))
            .join(
                predictions.select("athlete_id", pl.col(f"prior_{side}").alias("prior")),
                on="athlete_id",
                how="left",
            )
            .with_columns(pl.col("prior").fill_null(0.0))
        )
        shift = shares.group_by("team").agg(
            (
                (pl.col("rating").first() - (pl.col("prior") * pl.col("share")).sum())
                / pl.col("share").sum()
            ).alias(f"shift_{side}"),
        )
        sides.append(shares.select("team", "athlete_id", "poss").join(shift, on="team"))

    merged = (
        sides[0]
        .join(sides[1], on=["team", "athlete_id"], how="full", coalesce=True)
        .with_columns(
            (pl.col("poss").fill_null(0) + pl.col("poss_right").fill_null(0)).alias("total")
        )
        .sort("total", descending=True)
        .unique(subset=["athlete_id"], keep="first")
    )
    return (
        predictions.join(
            merged.select("athlete_id", "shift_off", "shift_def"), on="athlete_id", how="left"
        )
        .with_columns(
            (pl.col("prior_off") + pl.col("shift_off").fill_null(0.0)).alias("prior_off"),
            (pl.col("prior_def") + pl.col("shift_def").fill_null(0.0)).alias("prior_def"),
        )
        .select("athlete_id", "prior_off", "prior_def")
    )
