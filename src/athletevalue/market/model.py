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

Choosing the penalty by the same leave-one-school-out error that is then reported
makes that error optimistic: with pure-noise labels the best of four penalties can
beat a tier median by chance. The error used to decide whether the model is worth
using is therefore nested: for each held-out school the penalty is chosen by
leave-one-school-out error on the other schools only.
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
    """Leave-one-school-out mean absolute error in log dollars at the chosen penalty.

    Optimistic, because the penalty was chosen on it; see ``nested_log_mae``."""
    baseline_log_mae: float
    """The same error for a tier-median predictor."""
    cv_by_lambda: tuple[tuple[float, float], ...]
    nested_residuals: FloatArray
    """Absolute held-out residual of each label with the penalty chosen without its school."""

    @property
    def nested_log_mae(self) -> float:
        return float(self.nested_residuals.mean())

    def _matrix(self, X: FloatArray) -> FloatArray:
        standardized = (X - self.mean) / self.scale
        matrix: FloatArray = np.column_stack([np.ones(len(standardized)), standardized])
        return matrix

    def predict_log(self, X: FloatArray) -> FloatArray:
        result: FloatArray = self._matrix(X) @ self.coef
        return result

    def interval_log(self, X: FloatArray, level: float) -> tuple[FloatArray, FloatArray]:
        """CV+ lower and upper bounds in log dollars, one pair per row of *X*.

        Each tail uses ``(1 - level) / 2``. Barber et al. (2021) prove the interval
        formed with miscoverage ``a`` in each tail covers a new point with probability
        at least ``1 - 2a`` under exchangeability, so this choice guarantees at least
        *level*; the one-tail-each choice with ``a = 1 - level`` would guarantee only
        ``2 * level - 1``. Under exchangeability the interval is conservative: on
        synthetic data at level 0.8 it covers about 0.9. Labels are disclosed deals,
        which are not exchangeable with undisclosed ones, so the guarantee applies to
        players like those in the registry, not to every player priced.
        """
        matrix = self._matrix(X)
        loo = matrix @ self.fold_coef[self.fold_of_label].T  # rows: new points, cols: labels
        alpha = (1 - level) / 2
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
        nested_residuals=_nested_residuals(matrix, log_pay, fold_of_label, ridge_grid),
    )


def _nested_residuals(
    matrix: FloatArray,
    log_pay: FloatArray,
    fold_of_label: NDArray[np.int64],
    ridge_grid: tuple[float, ...],
) -> FloatArray:
    residuals = np.empty(len(log_pay))
    for k in np.unique(fold_of_label):
        held = fold_of_label == k
        train_x, train_y, train_folds = matrix[~held], log_pay[~held], fold_of_label[~held]
        inner_folds = np.unique(train_folds)

        def inner_error(
            lam: float,
            x: FloatArray = train_x,
            y: FloatArray = train_y,
            folds: NDArray[np.int64] = train_folds,
            groups: NDArray[np.int64] = inner_folds,
        ) -> float:
            errors = [
                np.abs(y[folds == j] - x[folds == j] @ _ridge(x[folds != j], y[folds != j], lam))
                for j in groups
            ]
            return float(np.concatenate(errors).mean())

        best = min(ridge_grid, key=inner_error) if len(inner_folds) > 1 else ridge_grid[0]
        residuals[held] = np.abs(log_pay[held] - matrix[held] @ _ridge(train_x, train_y, best))
    return residuals


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
