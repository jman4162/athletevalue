"""NCAA tournament bids and the money they bring a conference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import expit

from athletevalue.constants import EVEN_WIN_PCT

FloatArray = NDArray[np.float64]

TERMS = ("intercept", "win_pct", "power", "win_pct_x_power", "sos")


@dataclass(frozen=True)
class BidModel:
    """Logistic model of receiving a bid, given a team's record and who it played.

    Win percentage alone rates a .600 record in a one-bid league like a .600 record
    in a power league. ``sos``, the mean opponent record, is what the selection
    committee weighs against the record, and it enters centred on an even schedule
    so the other coefficients keep their meaning at ``sos = EVEN_WIN_PCT``.
    """

    coef: FloatArray
    n_obs: int

    def probability(
        self, win_pct: FloatArray | float, power: bool, sos: FloatArray | float = EVEN_WIN_PCT
    ) -> FloatArray:
        wp = np.asarray(win_pct, dtype=np.float64)
        strength = np.asarray(sos, dtype=np.float64) - EVEN_WIN_PCT
        p = float(power)
        c = dict(zip(TERMS, self.coef, strict=True))
        eta = (
            c["intercept"]
            + c["win_pct"] * wp
            + c["power"] * p
            + c["win_pct_x_power"] * wp * p
            + c["sos"] * strength
        )
        result: FloatArray = expit(eta)
        return result


def bid_sample(outcomes: pl.DataFrame, *, excluded_seasons: frozenset[int]) -> pl.DataFrame:
    """The team-seasons the bid model is fitted on and scored against."""
    return outcomes.filter(
        (pl.col("games") > 0)
        & ~pl.col("season").is_in(pl.Series(sorted(excluded_seasons), dtype=pl.Int64).implode())
    )


def _design(
    sample: pl.DataFrame, power_conferences: frozenset[str]
) -> tuple[FloatArray, FloatArray]:
    wp = (sample["wins"] / sample["games"]).cast(pl.Float64).to_numpy()
    power = (
        sample["conference"]
        .is_in(pl.Series(sorted(power_conferences)).implode())
        .cast(pl.Float64)
        .to_numpy()
    )
    sos = sample["sos"].cast(pl.Float64).fill_null(EVEN_WIN_PCT).to_numpy() - EVEN_WIN_PCT
    bid = sample["ncaa_bid"].cast(pl.Float64).to_numpy()
    return np.column_stack([np.ones_like(wp), wp, power, wp * power, sos]), bid


def _fit(X: FloatArray, bid: FloatArray, ridge: float) -> FloatArray:
    penalized = np.ones(X.shape[1])
    penalized[0] = 0.0  # the intercept carries the base rate, not a shrinkable effect

    def negative_log_likelihood(beta: FloatArray) -> tuple[float, FloatArray]:
        eta = X @ beta
        p = expit(eta)
        loss = float(np.sum(np.logaddexp(0, eta) - bid * eta))
        loss += 0.5 * ridge * float(np.sum(penalized * beta**2))
        return loss, X.T @ (p - bid) + ridge * penalized * beta

    result = minimize(negative_log_likelihood, np.zeros(X.shape[1]), jac=True, method="BFGS")
    return np.asarray(result.x, dtype=np.float64)


def fit_bid_model(
    outcomes: pl.DataFrame,
    *,
    power_conferences: frozenset[str],
    excluded_seasons: frozenset[int],
    ridge: float = 0.0,
) -> BidModel:
    sample = bid_sample(outcomes, excluded_seasons=excluded_seasons)
    X, bid = _design(sample, power_conferences)
    return BidModel(coef=_fit(X, bid, ridge), n_obs=sample.height)


def holdout_bid_predictions(
    outcomes: pl.DataFrame,
    *,
    power_conferences: frozenset[str],
    excluded_seasons: frozenset[int],
    ridge: float = 0.0,
) -> pl.DataFrame:
    """team, season, predicted, observed, with each season scored by the other seasons.

    The shipped model is fitted on every season, which is the most data. Scoring it
    on those same rows tests the logistic form and nothing else, so calibration is
    measured here instead: each season's teams are predicted by a model that never
    saw them.
    """
    sample = bid_sample(outcomes, excluded_seasons=excluded_seasons)
    seasons = sorted(sample["season"].unique().to_list())
    if len(seasons) < 2:
        raise ValueError("leave-one-season-out calibration needs at least two seasons")
    parts = []
    for season in seasons:
        held_out = sample.filter(pl.col("season") == season)
        rest = sample.filter(pl.col("season") != season)
        X_train, bid_train = _design(rest, power_conferences)
        model = BidModel(coef=_fit(X_train, bid_train, ridge), n_obs=rest.height)
        win_pct = (held_out["wins"] / held_out["games"]).cast(pl.Float64).to_numpy()
        power = held_out["conference"].is_in(sorted(power_conferences)).to_numpy()
        sos = held_out["sos"].cast(pl.Float64).fill_null(EVEN_WIN_PCT).to_numpy()
        predicted = np.where(
            power,
            model.probability(win_pct, True, sos),
            model.probability(win_pct, False, sos),
        )
        parts.append(
            held_out.select("team", "season").with_columns(
                pl.Series("predicted", predicted),
                held_out["ncaa_bid"].cast(pl.Float64).alias("observed"),
            )
        )
    return pl.concat(parts)


def units_per_bid(outcomes: pl.DataFrame, *, excluded_seasons: frozenset[int]) -> float:
    """Units the average bid earns, counting every team in the field equally.

    A conference earns one unit for each game a member plays except the final, which
    ``ncaa_units`` already excludes. Deep runs pull this above what a bubble team
    brings, so it describes the field rather than a marginal bid.
    """
    per_season = (
        outcomes.filter(
            ~pl.col("season").is_in(pl.Series(sorted(excluded_seasons), dtype=pl.Int64).implode())
        )
        .group_by("season")
        .agg(pl.col("ncaa_units").sum().alias("units"), pl.col("ncaa_bid").sum().alias("bids"))
        .filter(pl.col("bids") > 0)
    )
    if per_season.is_empty():
        raise ValueError("no seasons with tournament bids")
    return float((per_season["units"] / per_season["bids"]).mean())  # type: ignore[arg-type]


def marginal_units_per_bid(
    outcomes: pl.DataFrame,
    model: BidModel,
    *,
    power_conferences: frozenset[str],
    excluded_seasons: frozenset[int],
) -> float:
    """Units a bid earns for the teams whose bid is actually in doubt.

    A player's wins change program value through the bid probability, and
    ``dP/dwin_pct`` is proportional to ``p(1 - p)``. The teams that term weights are
    the ones a win moves in or out of the field, so weighting each bid team's units
    by ``p(1 - p)`` measures what such a bid brings: close to one game, against the
    field average of about two. No threshold is needed, and none is chosen.
    """
    sample = bid_sample(outcomes, excluded_seasons=excluded_seasons).filter(pl.col("ncaa_bid"))
    if sample.is_empty():
        raise ValueError("no seasons with tournament bids")
    win_pct = (sample["wins"] / sample["games"]).cast(pl.Float64).to_numpy()
    power = sample["conference"].is_in(sorted(power_conferences)).to_numpy()
    sos = sample["sos"].cast(pl.Float64).fill_null(EVEN_WIN_PCT).to_numpy()
    p = np.where(
        power, model.probability(win_pct, True, sos), model.probability(win_pct, False, sos)
    )
    weight = p * (1.0 - p)
    units = sample["ncaa_units"].cast(pl.Float64).to_numpy()
    total = float(weight.sum())
    if total <= 0.0:
        raise ValueError("the bid model separates every team; no bid is marginal")
    return float(np.dot(weight, units) / total)
