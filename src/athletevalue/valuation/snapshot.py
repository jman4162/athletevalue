"""Aggregate evidence for the documentation: gates, CV curves, rating precision, WAR, revenue.

The snapshot is the single source for every figure and quoted validation number in
the README and the methodology site. It holds arrays and summary statistics only:
no player names, athlete ids or person ids, so it can be committed and published.
Team-level arrays carry no team names either; nothing in the figures needs them.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl
from scipy.stats import norm

from athletevalue.assumptions.registry import AssumptionRegistry
from athletevalue.identity.names import normalize_name
from athletevalue.impact.cv import cv_with_folds, cv_with_prior, game_folds
from athletevalue.valuation.checks import ReturningPlayers, bid_calibration
from athletevalue.valuation.economy import EconomicsModel
from athletevalue.valuation.season import SeasonModel
from athletevalue.valuation.team import unit_terms
from athletevalue.valuation.validate import ValidationReport

SCHEMA_VERSION = 2
FORBIDDEN_KEYS = frozenset({"name", "athlete_id", "person_id", "player", "athlete_name"})
"""Keys that would identify a person; ``assert_anonymous`` rejects any of them."""

Json = dict[str, Any]


def gates_payload(report: ValidationReport) -> list[Json]:
    return [
        {
            "gate": gate.name,
            "value": gate.value,
            "low": gate.low,
            "high": gate.high,
            "passed": gate.passed,
            "detail": gate.detail,
        }
        for gate in report.gates
    ]


def cv_curves(model: SeasonModel, registry: AssumptionRegistry, *, seed: int = 0) -> Json:
    """Held-out error by penalty for no prior, a zero box prior, and the box prior.

    All three use the same game folds. Both prior curves rebuild the team adjustment
    from each fold's training games; the zero-box curve isolates what the adjustment
    alone contributes.
    """
    lambdas = registry.get("mbb.impact.lambda_grid").numbers()
    design = model.design
    folds = game_folds(
        design.groups,
        int(registry.get("mbb.impact.cv_folds").scalar()),
        np.random.default_rng(seed),
    )
    baseline_lam = registry.get("mbb.impact.ridge_lambda").scalar()
    adjust = registry.get("mbb.prior.team_adjustment").flag()
    curves: Json = {
        "lambdas": list(lambdas),
        "no_prior": list(cv_with_folds(design, lambdas, folds).mean_error),
    }
    zeros = pl.DataFrame(
        {
            "athlete_id": list(design.athlete_ids),
            "prior_off": np.zeros(design.n_players),
            "prior_def": np.zeros(design.n_players),
        }
    )
    _, zero_box = cv_with_prior(
        design,
        model.lineups,
        folds,
        lambdas,
        lam_baseline=baseline_lam,
        adjust_to_team=adjust,
        predictions_for=lambda _train: zeros,
    )
    curves["zero_box_team_adjusted"] = list(zero_box.mean_error)
    predictions_for = model.fold_predictions()
    if predictions_for is not None:
        _, box = cv_with_prior(
            design,
            model.lineups,
            folds,
            lambdas,
            lam_baseline=baseline_lam,
            adjust_to_team=adjust,
            predictions_for=predictions_for,
        )
        curves["box_team_adjusted"] = list(box.mean_error)
    return curves


def team_ratings_payload(model: SeasonModel, torvik: pl.DataFrame | None) -> Json:
    """Team net ratings with and without the prior, paired with Torvik where names match."""
    payload: Json = {
        "adj_net": model.teams["adj_net"].to_list(),
        "adj_net_no_prior": model.teams["adj_net_no_prior"].to_list(),
        "wins": model.teams["wins"].to_list(),
        "games": model.teams["games"].to_list(),
    }
    if torvik is not None:
        key = pl.col("team").map_elements(normalize_name, return_dtype=pl.Utf8).alias("key")
        joined = (
            model.teams.select("team", "adj_net", "adj_net_no_prior")
            .with_columns(key)
            .join(torvik.with_columns(key).select("key", "adj_em"), on="key", how="inner")
            .sort("adj_em")
        )
        payload["torvik"] = {
            "adj_em": joined["adj_em"].to_list(),
            "adj_net": joined["adj_net"].to_list(),
            "adj_net_no_prior": joined["adj_net_no_prior"].to_list(),
        }
    return payload


def rating_precision(model: SeasonModel) -> Json:
    """Possessions and posterior SE of net rating, with and without the prior, per player."""
    joined = (
        model.rapm.table.filter(~pl.col("pooled"))
        .select(
            "athlete_id", (pl.col("off_poss") + pl.col("def_poss")).alias("possessions"), "se_net"
        )
        .join(
            model.baseline.table.select("athlete_id", pl.col("se_net").alias("se_net_no_prior")),
            on="athlete_id",
            how="inner",
        )
        .sort("possessions", "se_net")
    )
    return {
        "possessions": joined["possessions"].to_list(),
        "se_net": joined["se_net"].to_list(),
        "se_net_no_prior": joined["se_net_no_prior"].to_list(),
    }


def war_payload(model: SeasonModel, players: pl.DataFrame) -> Json:
    """Replacement levels and starters' linear WAR under each definition, sorted."""
    starters = players.filter(pl.col("starter"))
    return {
        "headline_definition": model.replacement_definition,
        "replacement_levels": dict(model.replacement_levels),
        "starters": {
            definition: sorted(starters[f"war_{definition}"].to_list())
            for definition in model.replacement_levels
        },
    }


def returning_payload(result: ReturningPlayers | None) -> Json | None:
    if result is None:
        return None
    return {
        "earlier_season": result.earlier_season,
        "later_season": result.later_season,
        "n_players": result.n_players,
        "corr_without_prior": result.corr_without_prior,
        "corr_with_prior": result.corr_with_prior,
        "min_possessions_earlier": result.min_possessions_earlier,
        "min_possessions_later": result.min_possessions_later,
    }


def economics_payload(economics: EconomicsModel, registry: AssumptionRegistry) -> Json:
    revenue = economics.revenue
    level = registry.get("mbb.wins.interval_level").scalar()
    tails = [(1 - level) / 2, (1 + level) / 2]

    def summary(draws: np.ndarray) -> Json:
        return {
            "median": float(np.median(draws)),
            "interval": [float(v) for v in np.quantile(draws, tails)],
        }

    calibration = bid_calibration(economics)
    return {
        "first_season": revenue.first_season,
        "last_season": revenue.last_season,
        "n_obs": revenue.n_obs,
        "n_schools": revenue.n_schools,
        "interval_level": level,
        "win_effect_current": summary(revenue.win_effect_current()),
        "win_effect_two_season": summary(revenue.win_effect()),
        "bid_effect_current": summary(revenue.bid_effect_current()),
        "bid_effect_two_season": summary(revenue.bid_effect()),
        "draws": {
            "win_effect_current": sorted(revenue.win_effect_current().tolist()),
            "win_effect_two_season": sorted(revenue.win_effect().tolist()),
        },
        "units_per_bid": economics.units_per_bid,
        "units_per_bid_field": economics.units_per_bid_field,
        "unit_pv_factor": unit_terms(registry, economics.units_per_bid).pv_factor(),
        "bid_ridge_lambda": registry.get("economics.bid_ridge_lambda").scalar(),
        "bid_calibration": {
            "predicted": calibration["predicted"].to_list(),
            "observed": calibration["observed"].to_list(),
            "n": calibration["n"].to_list(),
        },
    }


def normal_interval_coverage(
    estimates: np.ndarray, standard_errors: np.ndarray, truth: np.ndarray, level: float
) -> float:
    """Share of *truth* inside ``estimate ± z * se`` at *level*."""
    z = float(norm.ppf((1 + level) / 2))
    inside = np.abs(estimates - truth) <= z * standard_errors
    return float(np.mean(inside))


def round_floats(value: Any, digits: int) -> Any:
    """Round floats to *digits* significant digits, so bit-level noise stays out of diffs."""
    if isinstance(value, float):
        if value == 0 or not math.isfinite(value):
            return value if math.isfinite(value) else None
        return float(f"{value:.{digits}g}")
    if isinstance(value, dict):
        return {key: round_floats(item, digits) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [round_floats(item, digits) for item in value]
    if isinstance(value, np.floating):
        return round_floats(float(value), digits)
    if isinstance(value, np.integer):
        return int(value)
    return value


def assert_anonymous(value: Any, *, forbidden_strings: frozenset[str] = frozenset()) -> None:
    """Raise if *value* has an identifying key, or contains any of *forbidden_strings*."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key in FORBIDDEN_KEYS:
                raise ValueError(f"snapshot has identifying key {key!r}")
            assert_anonymous(item, forbidden_strings=forbidden_strings)
    elif isinstance(value, list):
        for item in value:
            assert_anonymous(item, forbidden_strings=forbidden_strings)
    elif isinstance(value, str) and forbidden_strings:
        for text in forbidden_strings:
            if text and text in value:
                raise ValueError(f"snapshot text contains {text!r}")
