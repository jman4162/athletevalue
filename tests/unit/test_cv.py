from __future__ import annotations

import numpy as np

from athletevalue.impact.cv import cv_lambda, game_folds
from athletevalue.impact.design import build_design
from tests.fixtures.synthetic import POSSESSION_SD, make_season


def test_folds_never_split_a_game():
    groups = np.array([0, 0, 1, 1, 1, 2, 3, 3, 4])
    folds = game_folds(groups, 3, np.random.default_rng(0))
    for game in np.unique(groups):
        assert np.unique(folds[groups == game]).size == 1
    assert set(folds) <= {0, 1, 2}


def test_cv_picks_the_grid_point_nearest_the_bayes_optimal_penalty():
    """Ratings ~ N(0, tau^2) with noise sigma^2 make sigma^2 / tau^2 the optimal penalty."""
    rating_sd = 3.0
    season = make_season(n_teams=8, games_per_pair=12, rating_sd=rating_sd, seed=11)
    design = build_design(season.lineups, min_possessions=0)
    grid = (0.01, 30.0, 300.0, 3000.0, 1e8)
    result = cv_lambda(design, grid, n_folds=4, rng=np.random.default_rng(0))
    optimal = POSSESSION_SD**2 / rating_sd**2
    nearest = min(grid, key=lambda lam: abs(np.log(lam) - np.log(optimal)))
    assert result.best == nearest
    assert result.one_se >= result.best
