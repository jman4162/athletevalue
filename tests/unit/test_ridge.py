from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from athletevalue.impact.design import build_design
from athletevalue.impact.rapm import fit_rapm
from athletevalue.impact.ridge import RidgeError, fit_ridge, solve_penalized
from tests.fixtures.synthetic import make_season


def _truth(season, design):
    beta = np.zeros(design.n_columns)
    for k, athlete in enumerate(design.athlete_ids):
        beta[design.off_col(k)] = season.off[athlete]
        beta[design.def_col(k)] = season.defense[athlete]
    beta[design.home_col] = season.home
    beta[design.intercept_col] = season.intercept
    return beta


def test_recovers_known_ratings_with_ample_data():
    season = make_season(n_teams=8, games_per_pair=8, noise_scale=0.3, seed=1)
    design = build_design(season.lineups, min_possessions=0)
    fit = fit_ridge(design.X, design.y, design.w, 1.0, design.penalized)
    truth = _truth(season, design)
    p = design.n_players
    corr = np.corrcoef(fit.beta[: 2 * p], truth[: 2 * p])[0, 1]
    assert corr > 0.95
    assert fit.beta[design.home_col] == pytest.approx(season.home, abs=0.5)
    # Offense and defense sums identify the intercept only up to the mean rating.
    assert fit.beta[design.intercept_col] == pytest.approx(season.intercept, abs=2.0)


def test_penalty_shrinks_player_coefficients():
    season = make_season(seed=2)
    design = build_design(season.lineups, min_possessions=0)
    loose = fit_ridge(design.X, design.y, design.w, 1.0, design.penalized, compute_se=False)
    tight = fit_ridge(design.X, design.y, design.w, 1e7, design.penalized, compute_se=False)
    p = 2 * design.n_players
    assert np.linalg.norm(tight.beta[:p]) < 0.01 * np.linalg.norm(loose.beta[:p])


def test_weights_equal_row_duplication():
    rng = np.random.default_rng(3)
    X = sp.csr_matrix(rng.normal(size=(40, 5)))
    y = rng.normal(size=40)
    counts = rng.integers(1, 4, size=40)
    penalized = np.array([True, True, True, True, False])
    weighted = fit_ridge(X, y, counts.astype(float), 2.0, penalized, compute_se=False)
    dup_index = np.repeat(np.arange(40), counts)
    duplicated = fit_ridge(
        X[dup_index], y[dup_index], np.ones(dup_index.size), 2.0, penalized, compute_se=False
    )
    np.testing.assert_allclose(weighted.beta, duplicated.beta, atol=1e-10)


def test_row_and_column_permutation_invariance():
    rng = np.random.default_rng(4)
    X = sp.csr_matrix(rng.normal(size=(60, 6)))
    y = rng.normal(size=60)
    w = rng.uniform(1, 5, size=60)
    penalized = np.array([True] * 5 + [False])
    base = fit_ridge(X, y, w, 3.0, penalized)
    rows = rng.permutation(60)
    cols = rng.permutation(6)
    permuted = fit_ridge(X[rows][:, cols], y[rows], w[rows], 3.0, penalized[cols])
    np.testing.assert_allclose(permuted.beta, base.beta[cols], atol=1e-8)
    np.testing.assert_allclose(permuted.variance, base.variance[cols], atol=1e-8)


def test_prior_offset_is_ridge_toward_the_prior():
    rng = np.random.default_rng(5)
    X = sp.csr_matrix(rng.normal(size=(30, 4)))
    prior = np.array([2.0, -1.0, 0.5, 0.0])
    penalized = np.array([True, True, True, False])
    y = X @ prior
    fit = fit_ridge(X, y, np.ones(30), 50.0, penalized, offset=prior, compute_se=False)
    np.testing.assert_allclose(fit.beta, prior, atol=1e-10)


def test_offset_on_unpenalized_column_is_rejected():
    X = sp.csr_matrix(np.eye(3))
    with pytest.raises(RidgeError):
        fit_ridge(
            X,
            np.ones(3),
            np.ones(3),
            1.0,
            np.array([True, True, False]),
            offset=np.array([0, 0, 1.0]),
        )


def test_posterior_variance_matches_dense_inverse():
    rng = np.random.default_rng(6)
    Xd = rng.normal(size=(50, 6))
    X = sp.csr_matrix(Xd)
    y = rng.normal(size=50)
    w = rng.uniform(1, 3, size=50)
    penalized = np.array([True] * 5 + [False])
    lam = 4.0
    pairs = np.array([[0, 3], [5, 2]])
    fit = fit_ridge(X, y, w, lam, penalized, pairs=pairs)
    A = Xd.T @ np.diag(w) @ Xd + lam * np.diag(penalized.astype(float))
    inv = np.linalg.inv(A)
    df = 6 - lam * inv[penalized, penalized].sum()
    sigma2 = np.sum(w * (y - Xd @ fit.beta) ** 2) / (50 - df)
    assert fit.df == pytest.approx(df)
    assert fit.sigma2 == pytest.approx(sigma2)
    np.testing.assert_allclose(fit.variance, sigma2 * np.diag(inv))
    np.testing.assert_allclose(fit.pair_covariance, sigma2 * inv[[0, 2], [3, 5]])


def test_posterior_intervals_cover_ratings_drawn_from_the_prior():
    """If true ratings follow the ridge prior, 80% posterior intervals cover about 80%."""
    covered = total = 0
    lam = 1000.0
    for seed in range(12):
        season = make_season(
            n_teams=8, games_per_pair=3, rating_sd=float(np.sqrt(13225 / lam)), seed=100 + seed
        )
        result, _design, _ = fit_rapm(season.lineups, season=2026, lam=lam, min_possessions=0)
        table = result.table
        truth_net = np.array([season.off[a] + season.defense[a] for a in table["athlete_id"]])
        inside = np.abs(table["net"].to_numpy() - truth_net) <= 1.2816 * table["se_net"].to_numpy()
        covered += int(inside.sum())
        total += inside.size
    assert 0.72 <= covered / total <= 0.88


def test_fit_rapm_reports_pooled_players_and_scale():
    season = make_season(seed=7, non_d1_games=10, neutral_share=0.2)
    result, design, _fit = fit_rapm(season.lineups, season=2026, lam=1000.0, min_possessions=0)
    assert result.table.height == len(season.off)
    assert not result.table["pooled"].any()
    assert result.intercept == pytest.approx(105, abs=3)
    assert result.home_court == pytest.approx(3.0, abs=1.5)
    assert result.non_d1_net < -20
    assert 0 < result.df < design.n_columns


def test_work_buffer_holds_the_factor_and_matches_a_fresh_solve():

    rng = np.random.default_rng(3)
    A = rng.normal(size=(40, 6))
    gram = A.T @ A
    moment = A.T @ rng.normal(size=40)
    penalized = np.array([True] * 5 + [False])
    fresh, _ = solve_penalized(gram, moment, 2.0, penalized)
    work = np.empty_like(gram, order="F")
    reused, factor = solve_penalized(gram, moment, 2.0, penalized, work=work)
    np.testing.assert_array_equal(fresh, reused)
    assert np.shares_memory(factor, work)
    with pytest.raises(RidgeError, match="Fortran"):
        solve_penalized(gram, moment, 2.0, penalized, work=np.empty_like(gram, order="C"))
