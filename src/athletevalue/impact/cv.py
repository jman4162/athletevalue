"""Choosing the ridge penalty by cross-validation grouped on games.

Rows from one game share players, opponents and game state, so splitting a game
across folds leaks information. Folds therefore hold out whole games. Each
fold's Gram matrix is the full Gram minus the held-out part, so the expensive
product is computed once.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from athletevalue.impact.design import Design
from athletevalue.impact.ridge import solve_penalized, weighted_gram, weighted_moment


@dataclass(frozen=True)
class CvResult:
    lambdas: tuple[float, ...]
    mean_error: tuple[float, ...]
    """Possession-weighted held-out squared error, averaged over folds."""
    se_error: tuple[float, ...]
    best: float
    one_se: float
    """Largest lambda whose error is within one standard error of the best."""


def game_folds(
    groups: NDArray[np.int64], n_folds: int, rng: np.random.Generator
) -> NDArray[np.int64]:
    """Fold id per row; every row of a game gets the same fold."""
    games = np.unique(groups)
    shuffled = rng.permutation(games)
    fold_of_game = np.empty(games.max() + 1, dtype=np.int64)
    fold_of_game[shuffled] = np.arange(shuffled.size) % n_folds
    folds: NDArray[np.int64] = fold_of_game[groups]
    return folds


def cv_lambda(
    design: Design, lambdas: tuple[float, ...], *, n_folds: int, rng: np.random.Generator
) -> CvResult:
    folds = game_folds(design.groups, n_folds, rng)
    full_gram = weighted_gram(design.X, design.w)
    full_moment = weighted_moment(design.X, design.w, design.y)
    errors = np.zeros((n_folds, len(lambdas)))
    for fold in range(n_folds):
        held = folds == fold
        X_held = design.X[held]
        w_held = design.w[held]
        y_held = design.y[held]
        gram = full_gram - weighted_gram(X_held, w_held)
        moment = full_moment - weighted_moment(X_held, w_held, y_held)
        for k, lam in enumerate(lambdas):
            beta, _ = solve_penalized(gram, moment, lam, design.penalized)
            residual = y_held - X_held @ beta
            errors[fold, k] = np.sum(w_held * residual**2) / np.sum(w_held)
    mean = errors.mean(axis=0)
    se = errors.std(axis=0, ddof=1) / np.sqrt(n_folds)
    best_index = int(np.argmin(mean))
    threshold = mean[best_index] + se[best_index]
    one_se = max(lam for lam, err in zip(lambdas, mean, strict=True) if err <= threshold)
    return CvResult(
        lambdas=lambdas,
        mean_error=tuple(float(v) for v in mean),
        se_error=tuple(float(v) for v in se),
        best=lambdas[best_index],
        one_se=one_se,
    )
