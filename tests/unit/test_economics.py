from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from athletevalue.economics.panel import revenue_base, revenue_panel
from athletevalue.economics.program_value import ProgramContext, program_value_draws
from athletevalue.economics.revenue import InsufficientPanelError, RevenueModel, fit_revenue_model
from athletevalue.economics.tournament import BidModel, fit_bid_model, units_per_bid


def _synthetic_panel(seed: int = 0, n_schools: int = 60, seasons=range(2012, 2024)):
    rng = np.random.default_rng(seed)
    beta = {"wins": 0.01, "wins_lag": 0.005, "bid": 0.04, "bid_lag": 0.02}
    rows = []
    for s in range(n_schools):
        brand = rng.normal(15, 1)
        previous = None
        for t in seasons:
            wins = int(rng.integers(8, 30))
            bid = bool(rng.random() < (wins - 8) / 30)
            if previous is not None:
                log_rev = (
                    brand
                    + 0.03 * (t - 2012)
                    + beta["wins"] * wins
                    + beta["wins_lag"] * previous[0]
                    + beta["bid"] * bid
                    + beta["bid_lag"] * previous[1]
                    + rng.normal(0, 0.02)
                )
                rows.append(
                    {
                        "team": f"S{s}",
                        "season": t,
                        "conference": "X",
                        "games": 32,
                        "wins": wins,
                        "ncaa_bid": bid,
                        "rev_men": float(np.exp(log_rev)),
                        "exp_men": 1.0,
                        "allocated": False,
                        "wins_lag": previous[0],
                        "bid_lag": previous[1],
                        "has_lag": True,
                    }
                )
            previous = (wins, bid)
    return pl.DataFrame(rows), beta


def test_fixed_effects_recover_known_coefficients():
    panel, beta = _synthetic_panel()
    model = fit_revenue_model(
        panel, excluded_seasons=frozenset(), min_seasons=4, n_boot=30, rng=np.random.default_rng(1)
    )
    np.testing.assert_allclose(model.point, list(beta.values()), atol=0.004)
    assert np.quantile(model.win_effect(), 0.1) < 0.015 < np.quantile(model.win_effect(), 0.9)
    assert model.n_schools == 60


def test_allocated_and_excluded_rows_are_dropped():
    panel, _ = _synthetic_panel(n_schools=3, seasons=range(2012, 2015))
    allocated = panel.with_columns(pl.lit(True).alias("allocated"))
    with pytest.raises(InsufficientPanelError):
        fit_revenue_model(
            allocated,
            excluded_seasons=frozenset(),
            min_seasons=1,
            n_boot=2,
            rng=np.random.default_rng(0),
        )


def test_panel_builds_lags_and_follows_unitid_changes():
    outcomes = pl.DataFrame(
        {
            "team": ["A", "A", "A"],
            "season": [2020, 2021, 2023],
            "conference": ["X"] * 3,
            "games": [30, 30, 30],
            "wins": [10, 20, 25],
            "ncaa_bid": [False, True, True],
        }
    )
    eada = pl.DataFrame(
        {
            "unitid": [1, 1, 2],
            "season": [2020, 2021, 2023],
            "Sports": ["Basketball"] * 3,
            "REV_MEN": [5.0, 6.0, 7.0],
            "EXP_MEN": [5.0, 4.0, 3.0],
        }
    )
    crosswalk = pl.DataFrame({"team": ["A", "A"], "unitid": [1, 2]})
    panel = revenue_panel(outcomes, eada, crosswalk).sort("season")
    assert panel["allocated"].to_list() == [True, False, False]
    assert panel["wins_lag"].to_list() == [None, 10, 20]
    assert panel["has_lag"].to_list() == [False, True, False]
    assert revenue_base(panel, "A", through_season=2023, n_seasons=2) == pytest.approx(6.5)
    assert revenue_base(panel, "Z", through_season=2023, n_seasons=2) is None


def test_bid_model_and_units():
    rng = np.random.default_rng(3)
    wp = rng.uniform(0.2, 0.95, 4000)
    power = rng.random(4000) < 0.3
    eta = -10 + 14 * wp + power * (-5 + 10 * wp)
    bid = rng.random(4000) < 1 / (1 + np.exp(-eta))
    outcomes = pl.DataFrame(
        {
            "season": np.repeat(np.arange(2012, 2016), 1000),
            "games": 30,
            "wins": np.round(wp * 30).astype(int),
            "conference": np.where(power, "SEC", "Other"),
            "ncaa_bid": bid,
            "ncaa_games": np.where(bid, 2, 0),
        }
    )
    model = fit_bid_model(
        outcomes, power_conferences=frozenset({"SEC"}), excluded_seasons=frozenset()
    )
    assert (
        model.probability(0.9, True) > model.probability(0.9, False) > model.probability(0.5, False)
    )
    per_season = outcomes.group_by("season").agg(
        pl.col("ncaa_games").sum(), pl.col("ncaa_bid").sum()
    )
    expected = float(((per_season["ncaa_games"] - 2) / per_season["ncaa_bid"]).mean())
    assert units_per_bid(outcomes, excluded_seasons=frozenset()) == pytest.approx(expected)


def test_program_value_scales_with_wins_and_revenue():
    revenue = RevenueModel(
        point=np.zeros(4),
        draws=np.tile([0.01, 0.0, 0.05, 0.0], (10, 1)),
        n_obs=1,
        n_schools=1,
        first_season=2012,
        last_season=2025,
    )
    bids = BidModel(coef=np.array([-10.0, 15.0, 0.0, 0.0]), n_obs=1)
    context = ProgramContext(
        team="T", games=30, wins=20, power=False, revenue_base=10e6, conference_members=10
    )
    kwargs = {
        "unit_value": 2e6,
        "units_per_bid": 2.0,
        "share_multiplier": 1.0,
        "rng": np.random.default_rng(0),
    }
    zero = program_value_draws(np.zeros(100), context, revenue, bids, **kwargs)
    assert np.all(zero.total == 0)
    two = program_value_draws(np.full(100, 2.0), context, revenue, bids, **kwargs)
    np.testing.assert_allclose(two.win_revenue, 2 * 0.01 * 10e6)
    np.testing.assert_allclose(two.win_revenue_next, 0.0)
    np.testing.assert_allclose(two.annual, two.two_season)  # no lagged effect in this model
    delta = bids.probability(20 / 30, False) - bids.probability(18 / 30, False)
    np.testing.assert_allclose(two.bid_revenue, delta * 0.05 * 10e6)
    np.testing.assert_allclose(two.tournament_units, delta * 2.0 * 2e6 / 10)
