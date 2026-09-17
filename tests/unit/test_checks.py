"""Checks above the rating layer, on synthetic seasons."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from athletevalue.api import Session
from athletevalue.assumptions.registry import AssumptionRegistry
from athletevalue.economics.revenue import RevenueModel
from athletevalue.economics.tournament import BidModel, holdout_bid_predictions
from athletevalue.sources.cache import ArtifactCache
from athletevalue.valuation.checks import (
    bid_calibration,
    player_war_table,
    returning_players,
    team_war_table,
)
from athletevalue.valuation.economy import EconomicsModel
from athletevalue.valuation.validate import extended_gates


@pytest.fixture(scope="module")
def seasons(synthetic_cache_dir, one_prior_season):
    session = Session(
        cache=ArtifactCache(synthetic_cache_dir, offline=True),
        registry=AssumptionRegistry.load(extra_paths=(one_prior_season,)),
    )
    return session, session.fit_season(2041, prior=False), session.fit_season(2042)


def test_person_ids_link_the_synthetic_seasons(seasons):
    _, earlier, later = seasons
    assert earlier.rapm.table["person_id"].null_count() == 0
    result = returning_players(earlier, later, min_possessions_earlier=1, min_possessions_later=1)
    assert result is not None
    assert result.n_players > 30
    assert -1 <= result.corr_without_prior <= 1
    # Without a prior in the earlier fit, both correlations score the same ratings.
    assert result.gain == pytest.approx(0.0)


def test_returning_players_needs_person_ids(seasons):
    _, earlier, later = seasons
    stripped = earlier.baseline.table.with_columns(pl.lit(None, dtype=pl.Utf8).alias("person_id"))
    from dataclasses import replace

    unlinked = replace(earlier, baseline=replace(earlier.baseline, table=stripped))
    assert (
        returning_players(unlinked, later, min_possessions_earlier=1, min_possessions_later=1)
        is None
    )


def test_player_and_team_war_tables(seasons):
    _, _, later = seasons
    players = player_war_table(later)
    definitions = {f"war_{name}" for name in later.replacement_levels}
    assert definitions <= set(players.columns)
    assert players.group_by("team").agg(pl.col("starter").sum())["starter"].to_list() == [5] * 8
    shares = players.group_by("team").agg(pl.col("possession_share").sum())["possession_share"]
    np.testing.assert_allclose(shares.to_numpy(), 5.0, rtol=1e-9)
    teams = team_war_table(later, players)
    assert teams.height == 8
    assert np.corrcoef(teams["war"], teams["wins"])[0, 1] > 0.5


def _economics(teams: list[str]) -> EconomicsModel:
    rng = np.random.default_rng(0)
    n = 400
    win_pct = rng.uniform(0.2, 0.9, n)
    games = np.full(n, 30)
    power = rng.random(n) < 0.3
    sos = rng.uniform(0.4, 0.6, n)
    bids = BidModel(coef=np.array([-10.0, 15.0, -6.0, 14.0, 2.0]), n_obs=n)
    probability = np.where(
        power, bids.probability(win_pct, True, sos), bids.probability(win_pct, False, sos)
    )
    bid = rng.random(n) < probability
    outcomes = pl.DataFrame(
        {
            "team": [teams[k % len(teams)] for k in range(n)],
            # Two seasons, because leave-one-season-out calibration needs one to hold out.
            "season": np.where(np.arange(n) % 2 == 0, 2015, 2016),
            "conference": np.where(power, "Big Ten", "WCC"),
            "games": games,
            "wins": np.round(win_pct * games).astype(int),
            "sos": sos,
            "ncaa_bid": bid,
            "ncaa_units": np.where(bid, 1, 0),
        }
    )
    holdout = holdout_bid_predictions(
        outcomes, power_conferences=frozenset({"Big Ten"}), excluded_seasons=frozenset()
    )
    return EconomicsModel(
        panel=pl.DataFrame(),
        outcomes=outcomes,
        revenue=RevenueModel(
            point=np.zeros(4),
            draws=np.column_stack([rng.normal(0.004, 0.002, 200), np.zeros((200, 3))]),
            n_obs=1,
            n_schools=1,
            first_season=2012,
            last_season=2025,
        ),
        bids=bids,
        bid_holdout=holdout,
        units_per_bid=1.0,
        units_per_bid_field=1.9,
        power_conferences=frozenset({"Big Ten"}),
        reference_seasons=3,
        sources=(),
    )


def test_bid_calibration_of_a_correct_model_is_close():
    table = bid_calibration(_economics(["A", "B"]))
    assert table.height == 10
    assert int(table["n"].sum()) == 400
    assert float((table["predicted"] - table["observed"]).abs().max()) < 0.2


def test_extended_gates_report_skips_and_values(seasons, registry):
    session, earlier, later = seasons
    gates, notes = extended_gates(later, session.registry)
    assert [g.name for g in gates] == ["margin_sd", "team_war_vs_wins_correlation"]
    assert any("Returning-player gate skipped" in n for n in notes)
    assert any("Revenue and bid gates skipped" in n for n in notes)

    economics = _economics(later.teams["team"].to_list())
    gates, notes = extended_gates(later, session.registry, previous=earlier, economics=economics)
    by_name = {g.name: g for g in gates}
    assert set(by_name) == {
        "margin_sd",
        "team_war_vs_wins_correlation",
        "returning_player_prior_gain",
        "revenue_win_effect_nonpositive_share",
        "bid_calibration_max_decile_gap",
    }
    assert 0.0 <= by_name["revenue_win_effect_nonpositive_share"].value < 0.1
    assert notes == ()
