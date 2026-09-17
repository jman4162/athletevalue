"""Choosing the ridge penalty by cross-validation grouped on games.

Rows from one game share players, opponents and game state, so splitting a game
across folds leaks information. Folds therefore hold out whole games. Each
fold's Gram matrix is the full Gram minus the held-out part, so the expensive
product is computed once.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
import scipy.sparse as sp
from numpy.typing import NDArray

from athletevalue.frames.possessions import LineupData
from athletevalue.impact.design import Design
from athletevalue.impact.prior import prior_offset, team_adjust
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
    design: Design,
    lambdas: tuple[float, ...],
    *,
    n_folds: int,
    rng: np.random.Generator,
    offset: NDArray[np.float64] | None = None,
) -> CvResult:
    folds = game_folds(design.groups, n_folds, rng)
    return cv_with_folds(design, lambdas, folds, offset=offset)


def cv_with_folds(
    design: Design,
    lambdas: tuple[float, ...],
    folds: NDArray[np.int64],
    *,
    offset: NDArray[np.float64] | None = None,
) -> CvResult:
    """Grouped CV on given folds. With *offset*, ridge shrinks toward it instead of zero."""
    n_folds = int(folds.max()) + 1
    target = design.y if offset is None else design.y - design.X @ offset
    full_gram = weighted_gram(design.X, design.w)
    full_moment = weighted_moment(design.X, design.w, target)
    errors = np.zeros((n_folds, len(lambdas)))
    for fold in range(n_folds):
        held = folds == fold
        X_held = design.X[held]
        w_held = design.w[held]
        y_held = target[held]
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


def cv_prior_comparison(
    design: Design,
    lineups: LineupData,
    box_predictions: pl.DataFrame,
    folds: NDArray[np.int64],
    *,
    lam_without: float,
    lam_with: float,
    adjust_to_team: bool,
) -> tuple[float, float]:
    """Held-out error without and with the box prior, with no information from held-out games.

    The team adjustment needs team ratings, and team ratings from the full season
    would carry held-out games into the prior. Here each fold's adjustment uses a
    no-prior fit on that fold's training rows only.
    """
    n_folds = int(folds.max()) + 1
    full_gram = weighted_gram(design.X, design.w)
    raw_offset = prior_offset(design, box_predictions)
    without = with_prior = 0.0
    for fold in range(n_folds):
        held = folds == fold
        train = ~held
        X_held, w_held = design.X[held], design.w[held]
        gram = full_gram - weighted_gram(X_held, w_held)
        X_train, w_train = design.X[train], design.w[train]
        beta_none, _ = solve_penalized(
            gram, weighted_moment(X_train, w_train, design.y[train]), lam_without, design.penalized
        )
        without += _held_error(X_held, w_held, design.y[held], beta_none) / n_folds

        offset = raw_offset
        if adjust_to_team:
            adjusted = team_adjust(box_predictions, lineups, design, beta_none, rows=train)
            offset = prior_offset(design, adjusted)
        target = design.y - design.X @ offset
        gamma, _ = solve_penalized(
            gram, weighted_moment(X_train, w_train, target[train]), lam_with, design.penalized
        )
        with_prior += _held_error(X_held, w_held, target[held], gamma) / n_folds
    return without, with_prior


def _held_error(
    X: sp.csr_matrix, w: NDArray[np.float64], y: NDArray[np.float64], beta: NDArray[np.float64]
) -> float:
    residual = y - X @ beta
    return float(np.sum(w * residual**2) / np.sum(w))
