"""A fitted model of roster pay, with distribution-free intervals.

The target is log annual pay from disclosed deals. Features describe on-court value,
playing time and program resources. The model is ridge regression on standardized
features: with the few dozen labels a public registry can plausibly collect, a
linear model with a penalty chosen by cross-validation is the most that can be
fitted without overfitting, and its coefficients can be read.

Intervals use CV+ (Barber, Candès, Ramdas and Tibshirani, "Predictive inference
with the jackknife+", Annals of Statistics, 2021) with schools as folds: for a new
player, each labeled player i contributes the prediction of the model fitted
without i's school, plus and minus i's held-out residual. Quantiles of those
values give the interval. This needs no distributional assumption and stays valid
with small samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from athletevalue.constants import EPS

FloatArray = NDArray[np.float64]


class InsufficientLabelsError(ValueError):
    pass


@dataclass(frozen=True)
class MarketModel:
    features: tuple[str, ...]
    mean: FloatArray
    scale: FloatArray
    lam: float
    coef: FloatArray
    """Intercept then one coefficient per standardized feature, fitted on all labels."""
    fold_coef: FloatArray
    """Coefficients refitted without each school, one row per school."""
    fold_of_label: NDArray[np.int64]
    residuals: FloatArray
    """Absolute held-out residual of each label, log dollars."""
    n_labels: int
    n_schools: int
    cv_log_mae: float
    """Leave-one-school-out mean absolute error in log dollars at the chosen penalty."""
    baseline_log_mae: float
    """The same error for a tier-median predictor."""
    cv_by_lambda: tuple[tuple[float, float], ...]

    def _matrix(self, X: FloatArray) -> FloatArray:
        standardized = (X - self.mean) / self.scale
        matrix: FloatArray = np.column_stack([np.ones(len(standardized)), standardized])
        return matrix

    def predict_log(self, X: FloatArray) -> FloatArray:
        result: FloatArray = self._matrix(X) @ self.coef
        return result

    def interval_log(self, X: FloatArray, level: float) -> tuple[FloatArray, FloatArray]:
        """CV+ lower and upper bounds in log dollars, one pair per row of *X*.

        The residuals are absolute, so each bound uses the full miscoverage ``1 - level``,
        as in Barber et al. (2021), not half of it.
        """
        matrix = self._matrix(X)
        loo = matrix @ self.fold_coef[self.fold_of_label].T  # rows: new points, cols: labels
        alpha = 1 - level
        n = self.n_labels
        lower_rank = max(int(np.floor(alpha * (n + 1) + EPS)), 1)
        upper_rank = min(int(np.ceil((1 - alpha) * (n + 1) - EPS)), n)
        lower = np.sort(loo - self.residuals[None, :], axis=1)[:, lower_rank - 1]
        upper = np.sort(loo + self.residuals[None, :], axis=1)[:, upper_rank - 1]
        return lower, upper

    def draws_log(self, x: FloatArray, n: int, rng: np.random.Generator) -> FloatArray:
        """Draws for one feature row from the CV+ predictive set: a random label's
        held-out prediction plus or minus its residual."""
        loo = self._matrix(x[None, :]) @ self.fold_coef[self.fold_of_label].T
        pick = rng.integers(0, self.n_labels, n)
        sign = rng.choice(np.array([-1.0, 1.0]), n)
        result: FloatArray = loo[0, pick] + sign * self.residuals[pick]
        return result


def fit_market_model(
    X: FloatArray,
    log_pay: FloatArray,
    schools: NDArray[np.int64],
    tiers: NDArray[np.int64],
    *,
    features: tuple[str, ...],
    ridge_grid: tuple[float, ...],
    min_labels: int,
    min_schools: int,
) -> MarketModel:
    n_labels = len(log_pay)
    school_ids, fold_of_label = np.unique(schools, return_inverse=True)
    if n_labels < min_labels:
        raise InsufficientLabelsError(f"{n_labels} labeled player-seasons; {min_labels} needed")
    if len(school_ids) < min_schools:
        raise InsufficientLabelsError(
            f"{len(school_ids)} schools among labels; {min_schools} needed"
        )

    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale == 0] = 1.0
    matrix = np.column_stack([np.ones(n_labels), (X - mean) / scale])
    n_folds = len(school_ids)

    fits = []
    for lam in ridge_grid:
        fold_coef = np.vstack(
            [
                _ridge(matrix[fold_of_label != k], log_pay[fold_of_label != k], lam)
                for k in range(n_folds)
            ]
        )
        held = np.einsum("ij,ij->i", matrix, fold_coef[fold_of_label])
        residual = np.abs(log_pay - held)
        fits.append((float(residual.mean()), lam, fold_coef, residual))
    cv = tuple((lam, mae) for mae, lam, _, _ in fits)
    _, lam, fold_coef, residual = min(fits, key=lambda fit: fit[0])
    return MarketModel(
        features=features,
        mean=mean,
        scale=scale,
        lam=lam,
        coef=_ridge(matrix, log_pay, lam),
        fold_coef=fold_coef,
        fold_of_label=fold_of_label.astype(np.int64),
        residuals=residual,
        n_labels=n_labels,
        n_schools=n_folds,
        cv_log_mae=float(residual.mean()),
        baseline_log_mae=_tier_median_error(log_pay, fold_of_label, tiers),
        cv_by_lambda=cv,
    )


def _ridge(matrix: FloatArray, y: FloatArray, lam: float) -> FloatArray:
    penalty = np.full(matrix.shape[1], lam)
    penalty[0] = 0.0
    coef: FloatArray = np.linalg.solve(matrix.T @ matrix + np.diag(penalty), matrix.T @ y)
    return coef


def _tier_median_error(
    log_pay: FloatArray, fold_of_label: NDArray[np.int64], tiers: NDArray[np.int64]
) -> float:
    """Leave-one-school-out error of predicting each label by its tier's median."""
    errors = np.empty(len(log_pay))
    for i in range(len(log_pay)):
        pool = (fold_of_label != fold_of_label[i]) & (tiers == tiers[i])
        if not pool.any():
            pool = fold_of_label != fold_of_label[i]
        errors[i] = abs(log_pay[i] - float(np.median(log_pay[pool])))
    return float(errors.mean())
