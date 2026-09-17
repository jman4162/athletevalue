from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from athletevalue.economics.panel import revenue_base, revenue_panel
from athletevalue.economics.program_value import (
    ProgramContext,
    UnitOverlapPolicy,
    UnitTerms,
    program_value_draws,
)
from athletevalue.economics.revenue import InsufficientPanelError, RevenueModel, fit_revenue_model
from athletevalue.economics.tournament import (
    TERMS,
    BidModel,
    fit_bid_model,
    holdout_bid_predictions,
    marginal_units_per_bid,
    units_per_bid,
)


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


def _bid_outcomes(n: int = 4000, seed: int = 3, deep_runs: bool = False) -> pl.DataFrame:
    """Team-seasons whose bids follow a known logistic curve in record and schedule."""
    rng = np.random.default_rng(seed)
    wp = rng.uniform(0.2, 0.95, n)
    power = rng.random(n) < 0.3
    sos = rng.uniform(0.35, 0.65, n)
    eta = -10 + 14 * wp + power * (-5 + 10 * wp) + 8 * (sos - 0.5)
    bid = rng.random(n) < 1 / (1 + np.exp(-eta))
    probability = 1 / (1 + np.exp(-eta))
    if deep_runs:
        # Near-certain teams go deep; teams the model is unsure about play once.
        units = np.where(bid, np.where(probability > 0.9, 5, 1), 0)
    else:
        units = np.where(bid, 1, 0)
    return pl.DataFrame(
        {
            "team": [f"T{k}" for k in range(n)],
            "season": np.repeat(np.arange(2012, 2016), n // 4),
            "games": 30,
            "wins": np.round(wp * 30).astype(int),
            "conference": np.where(power, "SEC", "Other"),
            "sos": sos,
            "ncaa_bid": bid,
            "ncaa_units": units,
        }
    )


POWER = frozenset({"SEC"})


def test_bid_model_reads_record_and_schedule():
    outcomes = _bid_outcomes()
    model = fit_bid_model(outcomes, power_conferences=POWER, excluded_seasons=frozenset())
    assert (
        model.probability(0.9, True) > model.probability(0.9, False) > model.probability(0.5, False)
    )
    # The same record against a harder schedule is worth more to the committee.
    assert model.probability(0.7, False, 0.65) > model.probability(0.7, False, 0.35)
    assert dict(zip(TERMS, model.coef, strict=True))["sos"] > 0


def test_the_ridge_penalty_pulls_coefficients_toward_zero():
    outcomes = _bid_outcomes()
    free = fit_bid_model(outcomes, power_conferences=POWER, excluded_seasons=frozenset())
    heavy = fit_bid_model(
        outcomes, power_conferences=POWER, excluded_seasons=frozenset(), ridge=1000.0
    )
    assert np.abs(heavy.coef[1:]).sum() < np.abs(free.coef[1:]).sum()
    # The intercept is never penalised, so the base rate survives.
    assert np.abs(heavy.coef[0]) > 0.0


def test_every_season_is_scored_by_a_fit_that_never_saw_it():
    outcomes = _bid_outcomes()
    holdout = holdout_bid_predictions(
        outcomes, power_conferences=POWER, excluded_seasons=frozenset({2015})
    )
    assert sorted(holdout["season"].unique().to_list()) == [2012, 2013, 2014]
    assert holdout.height == outcomes.filter(pl.col("season") != 2015).height
    # A model fitted on the other seasons still calibrates: this is the real gate.
    assert holdout["predicted"].min() >= 0.0 and holdout["predicted"].max() <= 1.0
    gap = abs(float(holdout["predicted"].mean()) - float(holdout["observed"].mean()))
    assert gap < 0.05


def test_holdout_needs_more_than_one_season():
    one = _bid_outcomes().filter(pl.col("season") == 2012)
    with pytest.raises(ValueError, match="at least two seasons"):
        holdout_bid_predictions(one, power_conferences=POWER, excluded_seasons=frozenset())


def test_marginal_units_count_the_bids_a_win_actually_decides():
    """Deep runs belong to teams that were never in doubt, so they are not marginal."""
    outcomes = _bid_outcomes(deep_runs=True)
    model = fit_bid_model(outcomes, power_conferences=POWER, excluded_seasons=frozenset())
    marginal = marginal_units_per_bid(
        outcomes, model, power_conferences=POWER, excluded_seasons=frozenset()
    )
    field = units_per_bid(outcomes, excluded_seasons=frozenset())
    bubble_units = 1.0  # what a team that was in doubt plays for in this construction
    assert marginal < field
    assert abs(marginal - bubble_units) < abs(field - bubble_units)


def test_field_units_average_over_the_whole_bracket():
    outcomes = _bid_outcomes(deep_runs=True)
    per_season = outcomes.group_by("season").agg(
        pl.col("ncaa_units").sum(), pl.col("ncaa_bid").sum()
    )
    expected = float((per_season["ncaa_units"] / per_season["ncaa_bid"]).mean())
    assert units_per_bid(outcomes, excluded_seasons=frozenset()) == pytest.approx(expected)


def _terms(**overrides) -> UnitTerms:
    settings = {
        "unit_value": 2e6,
        "units_per_bid": 2.0,
        "share_multiplier": 1.0,
        "discount_rate": 0.0,
        "installments": 6,
        "first_year": 1,
        "overlap": UnitOverlapPolicy.NONE,
    }
    return UnitTerms(**{**settings, **overrides})


def test_present_value_factors_discount_each_instalment():
    undiscounted = _terms()
    assert undiscounted.pv_factor() == pytest.approx(1.0)
    assert undiscounted.first_installment_factor() == pytest.approx(1 / 6)

    discounted = _terms(discount_rate=0.05)
    expected = sum(1.05**-k for k in range(1, 7)) / 6
    assert discounted.pv_factor() == pytest.approx(expected)
    assert discounted.pv_factor() < 1.0
    assert discounted.first_installment_factor() == pytest.approx(1.05**-1 / 6)
    assert discounted.next_season_factor() == pytest.approx(1 / 1.05)


def _program(units: UnitTerms, war: float = 2.0):
    revenue = RevenueModel(
        point=np.zeros(4),
        draws=np.tile([0.01, 0.006, 0.05, 0.03], (10, 1)),
        n_obs=1,
        n_schools=1,
        first_season=2012,
        last_season=2025,
    )
    bids = BidModel(coef=np.array([-10.0, 15.0, 0.0, 0.0, 0.0]), n_obs=1)
    context = ProgramContext(
        team="T",
        games=30,
        wins=20,
        power=False,
        revenue_base=10e6,
        conference_members=10,
        sos=0.5,
    )
    draws = program_value_draws(
        np.full(100, war), context, revenue, bids, units=units, rng=np.random.default_rng(0)
    )
    delta = float(bids.probability(20 / 30, False) - bids.probability((20 - war) / 30, False))
    return draws, delta


def test_program_value_scales_with_wins_and_revenue():
    units = _terms()
    draws, delta = _program(units)
    np.testing.assert_allclose(draws.win_revenue_current, 2 * 0.01 * 10e6)
    np.testing.assert_allclose(draws.bid_revenue_current, delta * 0.05 * 10e6)
    np.testing.assert_allclose(draws.tournament_units, delta * 2.0 * 2e6 / 10)
    zero, _ = _program(units, war=0.0)
    assert np.all(zero.annual == 0)


def test_money_arriving_later_is_discounted():
    plain = _terms()
    discounted = _terms(discount_rate=0.05)
    later, _ = _program(discounted)
    now, _ = _program(plain)

    # Next season's carry-over and the unit instalments both arrive after the season.
    np.testing.assert_allclose(later.win_revenue_next, now.win_revenue_next / 1.05)
    np.testing.assert_allclose(later.bid_revenue_next, now.bid_revenue_next / 1.05)
    np.testing.assert_allclose(
        later.tournament_units, now.tournament_units * discounted.pv_factor()
    )
    # This season's revenue is not discounted at all.
    np.testing.assert_allclose(later.win_revenue_current, now.win_revenue_current)


def test_the_overlap_policy_changes_only_the_figure_it_should():
    none, _ = _program(_terms(overlap=UnitOverlapPolicy.NONE))
    default, _ = _program(_terms(overlap=UnitOverlapPolicy.NEXT_SEASON_INSTALLMENT))
    excluded, _ = _program(_terms(overlap=UnitOverlapPolicy.EXCLUDE_UNITS))

    # The component itself is always reported in full, whatever the policy counts.
    for draws in (none, default, excluded):
        np.testing.assert_allclose(draws.tournament_units, none.tournament_units)

    # Units are paid from the next fiscal year, so only the two-season figure overlaps.
    np.testing.assert_allclose(default.annual, none.annual)
    np.testing.assert_allclose(
        default.two_season, none.two_season - none.tournament_units_first_installment
    )

    # Dropping units removes them from both figures and nothing else.
    np.testing.assert_allclose(excluded.annual, none.annual - none.tournament_units)
    np.testing.assert_allclose(excluded.two_season, none.two_season - none.tournament_units)


def test_bootstrap_is_unchanged_by_categoricals_cast_earlier_in_the_process():
    panel, _ = _synthetic_panel()
    first = fit_revenue_model(
        panel, excluded_seasons=frozenset(), min_seasons=4, n_boot=5, rng=np.random.default_rng(2)
    )
    names = pl.Series([f"other {i}" for i in range(50)]).cast(pl.Categorical)
    again = fit_revenue_model(
        panel, excluded_seasons=frozenset(), min_seasons=4, n_boot=5, rng=np.random.default_rng(2)
    )
    assert names.len() == 50
    assert again.n_schools == first.n_schools == 60
    np.testing.assert_array_equal(again.draws, first.draws)
