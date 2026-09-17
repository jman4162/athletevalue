"""Choosing the ridge penalty by cross-validation grouped on games.

Rows from one game share players, opponents and game state, so splitting a game
across folds leaks information. Folds therefore hold out whole games. Each
fold's Gram matrix is the full Gram minus the held-out part, so the expensive
product is computed once. The held-out part stays sparse and is subtracted in
place, and every solve reuses one factor buffer, so a fold needs two dense Gram
copies beyond the full one rather than five.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import polars as pl
import scipy.sparse as sp
from numpy.typing import NDArray

from athletevalue.frames.possessions import LineupData
from athletevalue.impact.design import Design
from athletevalue.impact.prior import prior_offset, team_adjust
from athletevalue.impact.ridge import (
    FloatArray,
    solve_penalized,
    subtract_in_place,
    weighted_gram,
    weighted_gram_sparse,
    weighted_moment,
)


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
    """Grouped CV. *offset* must not depend on this season's outcomes; see ``cv_with_prior``."""
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
    gram, work = _buffers(full_gram)
    errors = np.zeros((n_folds, len(lambdas)))
    for fold in range(n_folds):
        held = folds == fold
        X_held = design.X[held]
        w_held = design.w[held]
        y_held = target[held]
        _held_out_gram(gram, full_gram, X_held, w_held)
        moment = full_moment - weighted_moment(X_held, w_held, y_held)
        for k, lam in enumerate(lambdas):
            beta, _ = solve_penalized(gram, moment, lam, design.penalized, work=work)
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


PredictionsFor = Callable[[NDArray[np.bool_]], pl.DataFrame]
"""Given a mask of training rows, returns box predictions (athlete_id, prior_off, prior_def)
computed from those rows' games only."""


def cv_with_prior(
    design: Design,
    lineups: LineupData,
    folds: NDArray[np.int64],
    lambdas: tuple[float, ...],
    *,
    lam_baseline: float,
    adjust_to_team: bool,
    predictions_for: PredictionsFor,
) -> tuple[float, CvResult]:
    """Held-out error without a prior (at *lam_baseline*) and with one at each of *lambdas*.

    Nothing from a held-out game reaches the prior it is scored against: box
    predictions come from *predictions_for* on the training rows, and the team
    adjustment uses a no-prior fit on those rows. A prior built from the full season
    would carry each held-out game's own outcome into its shrinkage target, which
    makes held-out error fall without bound as the penalty grows.
    """
    n_folds = int(folds.max()) + 1
    full_gram = weighted_gram(design.X, design.w)
    gram, work = _buffers(full_gram)
    without = 0.0
    errors = np.zeros((n_folds, len(lambdas)))
    for fold in range(n_folds):
        held = folds == fold
        train = ~held
        X_held, w_held = design.X[held], design.w[held]
        _held_out_gram(gram, full_gram, X_held, w_held)
        X_train, w_train = design.X[train], design.w[train]
        beta_none, _ = solve_penalized(
            gram,
            weighted_moment(X_train, w_train, design.y[train]),
            lam_baseline,
            design.penalized,
            work=work,
        )
        without += _held_error(X_held, w_held, design.y[held], beta_none) / n_folds

        predictions = predictions_for(train)
        if adjust_to_team:
            predictions = team_adjust(predictions, lineups, design, beta_none, rows=train)
        offset = prior_offset(design, predictions)
        target = design.y - design.X @ offset
        moment = weighted_moment(X_train, w_train, target[train])
        for k, lam in enumerate(lambdas):
            gamma, _ = solve_penalized(gram, moment, lam, design.penalized, work=work)
            errors[fold, k] = _held_error(X_held, w_held, target[held], gamma)
    mean = errors.mean(axis=0)
    se = errors.std(axis=0, ddof=1) / np.sqrt(n_folds) if n_folds > 1 else np.zeros_like(mean)
    best_index = int(np.argmin(mean))
    threshold = mean[best_index] + se[best_index]
    one_se = max(lam for lam, err in zip(lambdas, mean, strict=True) if err <= threshold)
    result = CvResult(
        lambdas=lambdas,
        mean_error=tuple(float(v) for v in mean),
        se_error=tuple(float(v) for v in se),
        best=lambdas[best_index],
        one_se=one_se,
    )
    return without, result


def cv_prior_comparison(
    design: Design,
    lineups: LineupData,
    box_predictions: pl.DataFrame | PredictionsFor,
    folds: NDArray[np.int64],
    *,
    lam_without: float,
    lam_with: float,
    adjust_to_team: bool,
) -> tuple[float, float]:
    """Held-out error without and with the box prior at one penalty each.

    *box_predictions* is either a callable as in ``cv_with_prior`` or a fixed frame.
    A fixed frame is only leak-free when it was not computed from this season's
    games (for example true ratings in a synthetic test).
    """
    predictions_for: PredictionsFor = (
        box_predictions if callable(box_predictions) else (lambda _train: box_predictions)
    )
    without, result = cv_with_prior(
        design,
        lineups,
        folds,
        (lam_with,),
        lam_baseline=lam_without,
        adjust_to_team=adjust_to_team,
        predictions_for=predictions_for,
    )
    return without, result.mean_error[0]


def _held_error(
    X: sp.csr_matrix, w: NDArray[np.float64], y: NDArray[np.float64], beta: NDArray[np.float64]
) -> float:
    residual = y - X @ beta
    return float(np.sum(w * residual**2) / np.sum(w))


def _buffers(full_gram: FloatArray) -> tuple[FloatArray, FloatArray]:
    """A training-Gram buffer and a factor buffer, both Fortran-ordered."""
    return np.empty_like(full_gram, order="F"), np.empty_like(full_gram, order="F")


def _held_out_gram(
    out: FloatArray, full_gram: FloatArray, X_held: sp.csr_matrix, w_held: FloatArray
) -> None:
    """Write the training rows' Gram, the full Gram minus the held-out rows', into *out*."""
    np.copyto(out, full_gram)
    subtract_in_place(out, weighted_gram_sparse(X_held, w_held))
