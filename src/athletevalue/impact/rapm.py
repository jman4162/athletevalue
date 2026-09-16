"""Regularized adjusted plus-minus for one season."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from athletevalue.frames.possessions import LineupData
from athletevalue.impact.cv import CvResult
from athletevalue.impact.design import Design, build_design
from athletevalue.impact.ridge import RidgeError, RidgeFit, fit_ridge


@dataclass(frozen=True)
class RapmResult:
    season: int
    table: pl.DataFrame
    """athlete_id, name, team, off_poss, def_poss, pooled, orapm, drapm, net,
    se_off, se_def, se_net. Pooled players carry the pooled coefficients."""
    intercept: float
    home_court: float
    """Home offense advantage per side, points per 100 possessions (``h``).

    The home team's net advantage is twice this."""
    pool_net: float
    non_d1_net: float
    lam: float
    sigma2: float
    df: float
    n_rows: int
    n_possessions: int
    cv: CvResult | None


def fit_rapm(
    lineups: LineupData,
    *,
    season: int,
    lam: float,
    min_possessions: float,
    cv: CvResult | None = None,
) -> tuple[RapmResult, Design, RidgeFit]:
    design = build_design(lineups, min_possessions=min_possessions)
    p = design.n_players
    pairs = np.column_stack(
        [
            np.append(np.arange(p), design.pool_off_col),
            np.append(np.arange(p, 2 * p), design.pool_def_col),
        ]
    ).astype(np.int64)
    fit = fit_ridge(design.X, design.y, design.w, lam, design.penalized, pairs=pairs)
    if fit.variance is None or fit.pair_covariance is None or fit.sigma2 is None or fit.df is None:
        raise RidgeError("standard errors were requested but not computed")
    if lam > 0 and not fit.df < design.n_columns - 1:
        # With a positive penalty the effective degrees of freedom must fall below the
        # column count. This catches a penalty that was silently dropped.
        raise RidgeError(f"penalty had no effect: df={fit.df:.1f} of {design.n_columns} columns")

    beta, var, pair_cov = fit.beta, fit.variance, fit.pair_covariance
    own_off = beta[:p]
    own_def = beta[p : 2 * p]
    coef = pl.DataFrame(
        {
            "athlete_id": list(design.athlete_ids),
            "orapm": own_off,
            "drapm": own_def,
            "se_off": np.sqrt(var[:p]),
            "se_def": np.sqrt(var[p : 2 * p]),
            "se_net": np.sqrt(var[:p] + var[p : 2 * p] + 2 * pair_cov[:p]),
        }
    )
    pool_off, pool_def = beta[design.pool_off_col], beta[design.pool_def_col]
    pool_se = (
        float(np.sqrt(var[design.pool_off_col])),
        float(np.sqrt(var[design.pool_def_col])),
        float(np.sqrt(var[design.pool_off_col] + var[design.pool_def_col] + 2 * pair_cov[p])),
    )
    table = (
        lineups.players.join(coef, on="athlete_id", how="left")
        .with_columns(pl.col("orapm").is_null().alias("pooled"))
        .with_columns(
            pl.col("orapm").fill_null(float(pool_off)),
            pl.col("drapm").fill_null(float(pool_def)),
            pl.col("se_off").fill_null(pool_se[0]),
            pl.col("se_def").fill_null(pool_se[1]),
            pl.col("se_net").fill_null(pool_se[2]),
        )
        .with_columns((pl.col("orapm") + pl.col("drapm")).alias("net"))
        .select(
            "athlete_id",
            "name",
            "team",
            "off_poss",
            "def_poss",
            "pooled",
            "orapm",
            "drapm",
            "net",
            "se_off",
            "se_def",
            "se_net",
        )
        .sort("net", descending=True)
    )
    non_d1_net = float(beta[design.non_d1_off_col] + beta[design.non_d1_def_col])
    result = RapmResult(
        season=season,
        table=table,
        intercept=float(beta[design.intercept_col]),
        home_court=float(beta[design.home_col]),
        pool_net=float(pool_off + pool_def),
        non_d1_net=non_d1_net,
        lam=lam,
        sigma2=fit.sigma2,
        df=fit.df,
        n_rows=fit.n_rows,
        n_possessions=int(design.w.sum()),
        cv=cv,
    )
    return result, design, fit
