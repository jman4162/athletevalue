from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from athletevalue.frames.box import BOX_STATS, player_box_totals
from athletevalue.impact.cv import cv_prior_comparison, game_folds
from athletevalue.impact.design import build_design
from athletevalue.impact.prior import (
    PriorTrainingError,
    box_rates,
    fit_box_prior,
    prior_offset,
    team_adjust,
)
from athletevalue.impact.rapm import fit_rapm
from athletevalue.impact.ridge import fit_ridge
from athletevalue.sports.mbb.loaders import training_seasons
from tests.fixtures.synthetic import make_season


def _synthetic_box(athletes, off, defense, seed):
    """Box totals whose points rate tracks offense and steal rate tracks defense."""
    rng = np.random.default_rng(seed)
    rows = []
    for athlete in athletes:
        poss = float(rng.integers(300, 2000))
        rates = dict.fromkeys(BOX_STATS, 5.0)
        rates["pts"] = 20 + 2.0 * off[athlete] + rng.normal(0, 0.5)
        rates["stl"] = 2 + 0.5 * defense[athlete] + rng.normal(0, 0.1)
        rows.append(
            {"athlete_id": athlete, "o_poss": poss, **{k: v * poss / 100 for k, v in rates.items()}}
        )
    return pl.DataFrame(rows)


def test_box_rates_add_league_average_pseudo_possessions():
    totals = pl.DataFrame(
        {"athlete_id": ["a"], "o_poss": [100.0], **{s: [0.0] for s in BOX_STATS}}
    ).with_columns(pl.lit(30.0).alias("pts"))
    league = np.full(len(BOX_STATS), 0.1)
    league[BOX_STATS.index("pts")] = 1.0
    rates = box_rates(totals, league, 100.0)
    # 30 points in 100 possessions plus 100 pseudo-possessions at 1 point each.
    assert rates[0, BOX_STATS.index("pts")] == pytest.approx(65.0)
    assert rates[0, BOX_STATS.index("ast")] == pytest.approx(5.0)


def test_fit_box_prior_learns_the_box_relationship():
    training = []
    for season, seed in ((2022, 1), (2023, 2)):
        synthetic = make_season(n_teams=8, seed=seed)
        table = fit_rapm(synthetic.lineups, season=season, lam=1.0, min_possessions=0)[0].table
        box = _synthetic_box(table["athlete_id"].to_list(), synthetic.off, synthetic.defense, seed)
        training.append((season, table, box))
    model = fit_box_prior(training, pseudo_possessions=0.0, ridge=1e-6)
    assert model.train_seasons == (2022, 2023)
    # The target is a noisy RAPM fit, so in-sample R^2 is modest; recovery is tested below.
    assert model.r2_off > 0.3 and model.r2_def > 0.3

    test = make_season(n_teams=8, seed=9)
    athletes = sorted(test.off)
    predictions = model.predict(_synthetic_box(athletes, test.off, test.defense, 9))
    truth_off = np.array([test.off[a] for a in predictions["athlete_id"]])
    assert np.corrcoef(predictions["prior_off"].to_numpy(), truth_off)[0, 1] > 0.8


def test_fit_box_prior_needs_data():
    with pytest.raises(PriorTrainingError):
        fit_box_prior([], pseudo_possessions=100.0, ridge=1.0)


def test_prior_offset_fills_only_modelled_player_columns():
    season = make_season(seed=3)
    design = build_design(season.lineups, min_possessions=0)
    athlete = design.athlete_ids[4]
    predictions = pl.DataFrame(
        {"athlete_id": [athlete, "unknown"], "prior_off": [1.5, 9.0], "prior_def": [-0.5, 9.0]}
    )
    offset = prior_offset(design, predictions)
    assert offset[design.off_col(4)] == 1.5 and offset[design.def_col(4)] == -0.5
    assert np.count_nonzero(offset) == 2


def test_team_adjustment_matches_team_ratings():
    season = make_season(seed=4)
    design = build_design(season.lineups, min_possessions=0)
    fit = fit_ridge(design.X, design.y, design.w, 1000.0, design.penalized, compute_se=False)
    athletes = list(design.athlete_ids)
    predictions = pl.DataFrame(
        {"athlete_id": athletes, "prior_off": np.linspace(-3, 3, len(athletes)), "prior_def": 1.0}
    )
    adjusted = team_adjust(predictions, season.lineups, design, fit.beta)
    offset = prior_offset(design, adjusted)
    off_cols = design.offense_columns()
    team_rows = season.lineups.rows["offense_team"] == "T00"
    w = design.w[team_rows.to_numpy()]
    X = design.X[team_rows.to_numpy()][:, off_cols]
    fitted = np.sum(w * (X @ fit.beta[off_cols])) / w.sum()
    from_prior = np.sum(w * (X @ offset[off_cols])) / w.sum()
    assert from_prior == pytest.approx(fitted, abs=1e-8)


def test_true_ratings_as_prior_lower_held_out_error():
    season = make_season(n_teams=8, games_per_pair=3, seed=5)
    design = build_design(season.lineups, min_possessions=0)
    truth = pl.DataFrame(
        {
            "athlete_id": list(design.athlete_ids),
            "prior_off": [season.off[a] for a in design.athlete_ids],
            "prior_def": [season.defense[a] for a in design.athlete_ids],
        }
    )
    folds = game_folds(design.groups, 4, np.random.default_rng(0))
    without, with_prior = cv_prior_comparison(
        design,
        season.lineups,
        truth,
        folds,
        lam_without=1000.0,
        lam_with=3000.0,
        adjust_to_team=False,
    )
    assert with_prior < without
    _, adjusted = cv_prior_comparison(
        design,
        season.lineups,
        truth,
        folds,
        lam_without=1000.0,
        lam_with=3000.0,
        adjust_to_team=True,
    )
    assert np.isfinite(adjusted)


def test_rapm_reports_prior_and_shrinks_toward_it():
    season = make_season(seed=6)
    athletes = sorted(season.off)
    prior = pl.DataFrame({"athlete_id": athletes, "prior_off": 5.0, "prior_def": 5.0})
    result, _, _ = fit_rapm(season.lineups, season=2026, lam=1e8, min_possessions=0, prior=prior)
    assert result.prior == "box"
    np.testing.assert_allclose(result.table["net"].to_numpy(), 10.0, atol=0.05)
    np.testing.assert_allclose(result.table["prior_net"].to_numpy(), 10.0)


def test_training_seasons_are_strictly_earlier():
    available = list(range(2011, 2027))
    assert training_seasons(2026, 4, available) == [2022, 2023, 2024, 2025]
    assert training_seasons(2025, 4, available) == [2021, 2022, 2023, 2024]
    assert training_seasons(2012, 4, available) == [2011]
    assert training_seasons(2011, 2, available) == []
    for target in available:
        assert all(s < target for s in training_seasons(target, 4, available))


def test_cv_with_prior_only_sees_training_rows():
    from athletevalue.impact.cv import cv_with_prior

    season = make_season(n_teams=6, games_per_pair=2, seed=8)
    design = build_design(season.lineups, min_possessions=0)
    folds = game_folds(design.groups, 3, np.random.default_rng(0))
    seen: list[np.ndarray] = []

    def predictions_for(train):
        seen.append(train.copy())
        return pl.DataFrame(
            {"athlete_id": list(design.athlete_ids), "prior_off": 0.0, "prior_def": 0.0}
        )

    without, result = cv_with_prior(
        design,
        season.lineups,
        folds,
        (1000.0,),
        lam_baseline=1000.0,
        adjust_to_team=False,
        predictions_for=predictions_for,
    )
    assert len(seen) == 3
    for fold, mask in enumerate(seen):
        assert not mask[folds == fold].any()  # held-out rows are never in the training mask
        assert mask[folds != fold].all()
    assert result.mean_error[0] == pytest.approx(without)  # a zero prior is no prior


def test_player_box_totals_sum_games():
    box = pl.DataFrame(
        {
            "player_id": ["1", "1", "2"],
            "o_poss": [10, 20, 5],
            **{s: [1.0, 2.0, 3.0] for s in BOX_STATS},
        }
    )
    totals = player_box_totals(box)
    assert totals.filter(pl.col("athlete_id") == "1")["o_poss"][0] == 30
    assert totals.filter(pl.col("athlete_id") == "1")["pts"][0] == 3.0
