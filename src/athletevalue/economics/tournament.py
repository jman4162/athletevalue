"""NCAA tournament bids and the money they bring a conference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import expit

FloatArray = NDArray[np.float64]

TERMS = ("intercept", "win_pct", "power", "win_pct_x_power")


@dataclass(frozen=True)
class BidModel:
    """Logistic model of receiving a bid given win percentage and power-conference membership."""

    coef: FloatArray
    n_obs: int

    def probability(self, win_pct: FloatArray | float, power: bool) -> FloatArray:
        wp = np.asarray(win_pct, dtype=np.float64)
        p = float(power)
        c = dict(zip(TERMS, self.coef, strict=True))
        eta = c["intercept"] + c["win_pct"] * wp + c["power"] * p + c["win_pct_x_power"] * wp * p
        result: FloatArray = expit(eta)
        return result


def fit_bid_model(
    outcomes: pl.DataFrame, *, power_conferences: frozenset[str], excluded_seasons: frozenset[int]
) -> BidModel:
    sample = outcomes.filter(
        (pl.col("games") > 0)
        & ~pl.col("season").is_in(pl.Series(sorted(excluded_seasons), dtype=pl.Int64).implode())
    )
    wp = (sample["wins"] / sample["games"]).cast(pl.Float64).to_numpy()
    power = (
        sample["conference"]
        .is_in(pl.Series(sorted(power_conferences)).implode())
        .cast(pl.Float64)
        .to_numpy()
    )
    bid = sample["ncaa_bid"].cast(pl.Float64).to_numpy()
    X = np.column_stack([np.ones_like(wp), wp, power, wp * power])

    def negative_log_likelihood(beta: FloatArray) -> tuple[float, FloatArray]:
        eta = X @ beta
        p = expit(eta)
        loss = float(np.sum(np.logaddexp(0, eta) - bid * eta))
        return loss, X.T @ (p - bid)

    result = minimize(negative_log_likelihood, np.zeros(X.shape[1]), jac=True, method="BFGS")
    return BidModel(coef=np.asarray(result.x, dtype=np.float64), n_obs=sample.height)


def units_per_bid(outcomes: pl.DataFrame, *, excluded_seasons: frozenset[int]) -> float:
    """Average tournament units a bid earns.

    A conference earns one unit for each game a member plays, except the
    championship game. Two team-games per season are therefore not units.
    """
    per_season = (
        outcomes.filter(
            ~pl.col("season").is_in(pl.Series(sorted(excluded_seasons), dtype=pl.Int64).implode())
        )
        .group_by("season")
        .agg(pl.col("ncaa_games").sum().alias("games"), pl.col("ncaa_bid").sum().alias("bids"))
        .filter(pl.col("bids") > 0)
    )
    if per_season.is_empty():
        raise ValueError("no seasons with tournament bids")
    championship_team_games = 2
    units = (per_season["games"] - championship_team_games) / per_season["bids"]
    return float(units.mean())  # type: ignore[arg-type]
