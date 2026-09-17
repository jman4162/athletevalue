from __future__ import annotations

import numpy as np
import pytest

from athletevalue.market.model import InsufficientLabelsError, fit_market_model


def _data(n=240, n_schools=24, noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    schools = rng.integers(0, n_schools, n)
    tiers = schools % 3
    log_pay = 12.0 + 0.8 * X[:, 0] - 0.4 * X[:, 1] - 0.5 * tiers + rng.normal(0, noise, n)
    return X, log_pay, schools, tiers


def test_recovers_signal_and_beats_tier_median():
    X, y, schools, tiers = _data()
    model = fit_market_model(
        X,
        y,
        schools,
        tiers,
        features=("a", "b", "c"),
        ridge_grid=(0.1, 1.0, 10.0),
        min_labels=40,
        min_schools=10,
    )
    slopes = model.coef[1:] / model.scale
    assert slopes[0] == pytest.approx(0.8, abs=0.1)
    assert slopes[1] == pytest.approx(-0.4, abs=0.1)
    assert model.cv_log_mae < model.baseline_log_mae
    assert model.nested_log_mae < model.baseline_log_mae
    assert model.lam == min(model.cv_by_lambda, key=lambda item: item[1])[0]
    assert model.n_schools == 24


def test_cv_plus_intervals_cover_new_points():
    X, y, schools, tiers = _data(seed=1)
    model = fit_market_model(
        X,
        y,
        schools,
        tiers,
        features=("a", "b", "c"),
        ridge_grid=(1.0,),
        min_labels=40,
        min_schools=10,
    )
    X_new, y_new, _, _ = _data(n=2000, seed=2)
    lower, upper = model.interval_log(X_new, 0.8)
    coverage = np.mean((y_new >= lower) & (y_new <= upper))
    assert coverage >= 0.75
    draws = model.draws_log(X_new[0], 5000, np.random.default_rng(0))
    assert lower[0] - 1e-9 <= np.median(draws) <= upper[0] + 1e-9


def test_minimums_are_enforced():
    X, y, schools, tiers = _data(n=30)
    with pytest.raises(InsufficientLabelsError):
        fit_market_model(
            X,
            y,
            schools,
            tiers,
            features=("a", "b", "c"),
            ridge_grid=(1.0,),
            min_labels=40,
            min_schools=10,
        )
    X, y, _, tiers = _data(n=100)
    with pytest.raises(InsufficientLabelsError):
        fit_market_model(
            X,
            y,
            np.zeros(100, dtype=int),
            tiers,
            features=("a", "b", "c"),
            ridge_grid=(1.0,),
            min_labels=40,
            min_schools=10,
        )


def test_nested_error_does_not_reward_noise():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(80, 6))
    schools = rng.integers(0, 12, 80)
    tiers = schools % 3
    y = 12.0 + rng.normal(0, 1.0, 80)
    model = fit_market_model(
        X,
        y,
        schools,
        tiers,
        features=tuple("abcdef"),
        ridge_grid=(0.1, 1.0, 10.0, 100.0),
        min_labels=40,
        min_schools=10,
    )
    assert model.nested_log_mae >= model.cv_log_mae
    assert model.nested_residuals.shape == (80,)
