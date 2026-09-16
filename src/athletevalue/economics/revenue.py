"""How a school's men's basketball revenue responds to winning.

Two-way fixed effects on log revenue:

    log R_st = a_s + g_t + b0 W_st + b1 W_s,t-1 + d0 Bid_st + d1 Bid_s,t-1 + e_st

School effects absorb brand, market size and conference; season effects absorb
league-wide growth. The identifying variation is a school winning more or less
than its own norm. Lagged terms follow Chung (2015), who finds athletic success
carries into later seasons' revenue. A win's revenue effect is ``b0 + b1`` times
the school's revenue.

Rows where revenue equals expense are dropped: those schools book the sport's
revenue as an allocation, so the number cannot respond to winning. Intervals come
from a bootstrap that resamples schools, since a school's seasons are correlated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

TERMS = ("wins", "wins_lag", "bid", "bid_lag")


class InsufficientPanelError(ValueError):
    pass


@dataclass(frozen=True)
class RevenueModel:
    point: FloatArray
    """Coefficients in ``TERMS`` order, log points."""
    draws: FloatArray
    """Bootstrap replicates, one row per replicate."""
    n_obs: int
    n_schools: int
    first_season: int
    last_season: int

    def win_effect(self) -> FloatArray:
        """Bootstrap draws of the log-revenue change per win, current plus next season."""
        return self._sum("wins", "wins_lag")

    def bid_effect(self) -> FloatArray:
        """Bootstrap draws of the log-revenue change from a bid, current plus next season."""
        return self._sum("bid", "bid_lag")

    def _sum(self, *terms: str) -> FloatArray:
        result: FloatArray = self.draws[:, [TERMS.index(t) for t in terms]].sum(axis=1)
        return result


def fit_sample(
    panel: pl.DataFrame, *, excluded_seasons: frozenset[int], min_seasons: int
) -> pl.DataFrame:
    usable = panel.filter(
        ~pl.col("allocated")
        & (pl.col("rev_men") > 0)
        & pl.col("has_lag")
        & ~pl.col("season").is_in(pl.Series(sorted(excluded_seasons), dtype=pl.Int64).implode())
        & ~(pl.col("season") - 1).is_in(
            pl.Series(sorted(excluded_seasons), dtype=pl.Int64).implode()
        )
    )
    counts = usable.group_by("team").agg(pl.len().alias("_n"))
    return usable.join(counts, on="team").filter(pl.col("_n") >= min_seasons).drop("_n")


def fit_revenue_model(
    panel: pl.DataFrame,
    *,
    excluded_seasons: frozenset[int],
    min_seasons: int,
    n_boot: int,
    rng: np.random.Generator,
) -> RevenueModel:
    sample = fit_sample(panel, excluded_seasons=excluded_seasons, min_seasons=min_seasons)
    if sample.height == 0 or sample["team"].n_unique() < 2 or sample["season"].n_unique() < 2:
        raise InsufficientPanelError("not enough non-allocated school-seasons to fit revenue")
    y = np.log(sample["rev_men"].to_numpy())
    X = np.column_stack(
        [
            sample["wins"].cast(pl.Float64).to_numpy(),
            sample["wins_lag"].cast(pl.Float64).to_numpy(),
            sample["ncaa_bid"].cast(pl.Float64).to_numpy(),
            sample["bid_lag"].cast(pl.Float64).to_numpy(),
        ]
    )
    school_codes = sample["team"].cast(pl.Categorical).to_physical().to_numpy().astype(np.int64)
    season_codes = np.unique(sample["season"].to_numpy(), return_inverse=True)[1]
    point = _two_way_fe(y, X, school_codes, season_codes)

    rows_by_school = [np.flatnonzero(school_codes == s) for s in range(school_codes.max() + 1)]
    n_schools = len(rows_by_school)
    draws = np.empty((n_boot, X.shape[1]))
    for b in range(n_boot):
        picked = rng.integers(0, n_schools, n_schools)
        index = np.concatenate([rows_by_school[s] for s in picked])
        relabel = np.concatenate([np.full(rows_by_school[s].size, k) for k, s in enumerate(picked)])
        draws[b] = _two_way_fe(y[index], X[index], relabel, season_codes[index])
    seasons = sample["season"]
    return RevenueModel(
        point=point,
        draws=draws,
        n_obs=sample.height,
        n_schools=n_schools,
        first_season=int(seasons.min()),  # type: ignore[arg-type]
        last_season=int(seasons.max()),  # type: ignore[arg-type]
    )


def _two_way_fe(
    y: FloatArray, X: FloatArray, schools: NDArray[np.int64], seasons: NDArray[np.int64]
) -> FloatArray:
    """Least squares with school and season dummies; returns the ``X`` coefficients."""
    _, school_index = np.unique(schools, return_inverse=True)
    _, season_index = np.unique(seasons, return_inverse=True)
    n = y.size
    school_dummies = np.zeros((n, school_index.max() + 1))
    school_dummies[np.arange(n), school_index] = 1
    season_dummies = np.zeros((n, season_index.max() + 1))
    season_dummies[np.arange(n), season_index] = 1
    design = np.hstack([X, school_dummies, season_dummies[:, 1:]])
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    result: FloatArray = coef[: X.shape[1]]
    return result
