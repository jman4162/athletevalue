"""Snapshot builders on synthetic seasons."""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from athletevalue.api import Session
from athletevalue.assumptions.registry import AssumptionRegistry
from athletevalue.sources.cache import ArtifactCache
from athletevalue.valuation.checks import player_war_table
from athletevalue.valuation.snapshot import (
    assert_anonymous,
    cv_curves,
    gates_payload,
    normal_interval_coverage,
    rating_precision,
    round_floats,
    team_ratings_payload,
    war_payload,
)


@pytest.fixture(scope="module")
def session_and_model(synthetic_cache_dir, one_prior_season):
    session = Session(
        cache=ArtifactCache(synthetic_cache_dir, offline=True),
        registry=AssumptionRegistry.load(extra_paths=(one_prior_season,)),
    )
    return session, session.fit_season(2042)


def test_cv_curves_share_folds_and_cover_the_grid(session_and_model):
    session, model = session_and_model
    curves = cv_curves(model, session.registry)
    grid = list(session.registry.get("mbb.impact.lambda_grid").numbers())
    assert curves["lambdas"] == grid
    for key in ("no_prior", "zero_box_team_adjusted", "box_team_adjusted"):
        assert len(curves[key]) == len(grid)
        assert all(np.isfinite(curves[key]))


def test_payloads_hold_no_identifiers(session_and_model):
    session, model = session_and_model
    torvik = model.teams.select(pl.col("team"), (pl.col("adj_net") + 0.5).alias("adj_em"))
    report = session.validate(model, use_torvik=False)
    payload = {
        "gates": gates_payload(report),
        "teams": team_ratings_payload(model, torvik),
        "precision": rating_precision(model),
        "war": war_payload(model, player_war_table(model)),
    }
    names = frozenset(model.rapm.table["name"].to_list()) | frozenset(model.teams["team"].to_list())
    assert_anonymous(payload, forbidden_strings=names)
    assert len(payload["teams"]["torvik"]["adj_em"]) == model.teams.height
    assert payload["teams"]["torvik"]["adj_em"] == sorted(payload["teams"]["torvik"]["adj_em"])
    precision = payload["precision"]
    assert (
        len(precision["possessions"])
        == len(precision["se_net"])
        == len(precision["se_net_no_prior"])
    )
    assert set(payload["war"]["starters"]) == set(model.replacement_levels)
    json.dumps(round_floats(payload, 5))


NAMES = frozenset({"Jordan Example"})


def test_assert_anonymous_rejects_keys_and_names():
    with pytest.raises(ValueError, match="identifying key"):
        assert_anonymous({"rows": [{"name": "x"}]})
    with pytest.raises(ValueError, match="contains"):
        assert_anonymous(
            {"notes": ["Jordan Example scored"]}, forbidden_strings=frozenset({"Jordan Example"})
        )
    assert_anonymous({"notes": ["no names here"]}, forbidden_strings=frozenset({"Jordan Example"}))


def test_round_floats_uses_significant_digits_and_drops_non_finite():
    assert round_floats({"a": [1.23456789, np.float64(0.000123456)], "b": float("nan")}, 3) == {
        "a": [1.23, 0.000123],
        "b": None,
    }


def test_normal_interval_coverage():
    truth = np.zeros(4)
    assert normal_interval_coverage(np.array([0.1, 2.0, -0.5, 5.0]), np.ones(4), truth, 0.8) == 0.5
