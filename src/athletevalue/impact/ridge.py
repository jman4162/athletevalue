"""Weighted ridge regression with a prior mean and exact posterior variances.

Minimizes ``sum_r w_r (y_r - x_r beta)^2 + lambda * sum_k p_k (beta_k - b0_k)^2``
where ``p_k`` marks penalized columns. Substituting ``gamma = beta - b0`` gives
an ordinary ridge problem on ``y - X b0``, so a box-score prior needs no special
solver.

Rows here aggregate possessions, with ``y`` a rate per 100 possessions and ``w``
the possession count. Each row's variance is ``sigma^2 / w``, so
``sum w r^2 / (n_rows - df)`` estimates the per-possession ``sigma^2`` (scaled by
100^2) whatever the aggregation.

Standard errors are posterior: ``sqrt(sigma^2 * diag(A^-1))`` with
``A = X'WX + lambda P``. They are exact, from the Cholesky factor of ``A``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.linalg import lapack

FloatArray = NDArray[np.float64]


class RidgeError(RuntimeError):
    pass


@dataclass(frozen=True)
class RidgeFit:
    beta: FloatArray
    lam: float
    fitted: FloatArray
    sigma2: float | None
    df: float | None
    n_rows: int
    variance: FloatArray | None
    """Posterior variance of each coefficient, ``sigma^2 * diag(A^-1)``."""
    pair_covariance: FloatArray | None
    """Posterior covariance for each requested column pair."""


def weighted_gram(X: sp.csr_matrix, w: FloatArray) -> FloatArray:
    """Dense ``X' W X``."""
    weighted = sp.diags(w) @ X
    gram: FloatArray = (X.T @ weighted).toarray()
    return gram


def weighted_moment(X: sp.csr_matrix, w: FloatArray, y: FloatArray) -> FloatArray:
    """``X' W y``."""
    moment: FloatArray = X.T @ (w * y)
    return moment


def solve_penalized(
    gram: FloatArray, moment: FloatArray, lam: float, penalized: NDArray[np.bool_]
) -> tuple[FloatArray, FloatArray]:
    """Solve ``(gram + lam P) gamma = moment``. Returns gamma and the Cholesky factor.

    An unpenalized column with no data (for example the pooled column when no
    player is pooled) would make the system singular. Its diagonal gets a unit
    entry, which pins its coefficient at zero without touching any other column.
    """
    if lam < 0:
        raise RidgeError(f"lambda must be non-negative, got {lam}")
    system = gram.copy()
    diagonal = np.diag_indices_from(system)
    system[diagonal] += lam * penalized + empty_unpenalized(gram, penalized)
    factor, info = lapack.dpotrf(system, lower=0, overwrite_a=1, clean=1)
    if info != 0:
        raise RidgeError(f"penalized Gram matrix is not positive definite (dpotrf info={info})")
    solution, info = lapack.dpotrs(factor, moment, lower=0)
    if info != 0:
        raise RidgeError(f"triangular solve failed (dpotrs info={info})")
    return np.asarray(solution, dtype=np.float64), np.asarray(factor, dtype=np.float64)


def empty_unpenalized(gram: FloatArray, penalized: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Unpenalized columns whose Gram diagonal is zero."""
    mask: NDArray[np.bool_] = (np.diag(gram) == 0) & ~penalized
    return mask


def fit_ridge(
    X: sp.csr_matrix,
    y: FloatArray,
    w: FloatArray,
    lam: float,
    penalized: NDArray[np.bool_],
    *,
    offset: FloatArray | None = None,
    compute_se: bool = True,
    pairs: NDArray[np.int64] | None = None,
    gram: FloatArray | None = None,
) -> RidgeFit:
    n_rows, n_columns = X.shape
    if y.shape != (n_rows,) or w.shape != (n_rows,):
        raise RidgeError("y and w must have one entry per row of X")
    if np.any(w <= 0):
        raise RidgeError("weights must be positive")
    prior = np.zeros(n_columns) if offset is None else offset
    if np.any(prior[~penalized] != 0):
        raise RidgeError("a prior mean is only meaningful for penalized columns")

    residual_target = y - X @ prior
    gram_matrix = weighted_gram(X, w) if gram is None else gram
    gamma, factor = solve_penalized(
        gram_matrix, weighted_moment(X, w, residual_target), lam, penalized
    )
    beta = gamma + prior
    fitted = np.asarray(X @ beta, dtype=np.float64)

    if not compute_se:
        return RidgeFit(beta, lam, fitted, None, None, n_rows, None, None)

    inverse, info = lapack.dpotri(factor, lower=0, overwrite_c=1)
    if info != 0:
        raise RidgeError(f"inverse from Cholesky factor failed (dpotri info={info})")
    diag_inverse = np.diag(inverse).copy()
    inactive = int(empty_unpenalized(gram_matrix, penalized).sum())
    df = float(n_columns - inactive - lam * diag_inverse[penalized].sum())
    residual = y - fitted
    dof = n_rows - df
    if dof <= 0:
        raise RidgeError("more effective parameters than rows")
    sigma2 = float(np.sum(w * residual**2) / dof)
    pair_cov = None
    if pairs is not None and len(pairs):
        low = np.minimum(pairs[:, 0], pairs[:, 1])
        high = np.maximum(pairs[:, 0], pairs[:, 1])
        pair_cov = sigma2 * inverse[low, high]  # dpotri fills the upper triangle
    return RidgeFit(
        beta=beta,
        lam=lam,
        fitted=fitted,
        sigma2=sigma2,
        df=df,
        n_rows=n_rows,
        variance=sigma2 * diag_inverse,
        pair_covariance=pair_cov,
    )
