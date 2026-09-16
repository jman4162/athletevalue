"""Pythagorean expectation for college basketball."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar


def win_pct(ortg: float, drtg: float, exponent: float) -> float:
    """Expected win percentage, ``ortg^x / (ortg^x + drtg^x)``."""
    ratio = (drtg / ortg) ** exponent
    return float(1 / (1 + ratio))


def margin_slope(ortg: float, drtg: float, exponent: float) -> float:
    """Change in expected win percentage per point per 100 possessions of net rating.

    The margin is added half to offense and half to defense. At ``ortg == drtg``
    this reduces to ``exponent / (4 * ortg)``.
    """
    x = exponent
    o, d = ortg, drtg
    denominator = (o**x + d**x) ** 2
    return float(0.5 * x * o ** (x - 1) * d ** (x - 1) * (o + d) / denominator)


def fit_exponent(
    ortg: NDArray[np.float64],
    drtg: NDArray[np.float64],
    win_fraction: NDArray[np.float64],
    games: NDArray[np.float64],
    *,
    bounds: tuple[float, float],
) -> float:
    """Exponent minimizing game-weighted squared error in win percentage."""

    def loss(x: float) -> float:
        predicted = 1 / (1 + (drtg / ortg) ** x)
        return float(np.sum(games * (win_fraction - predicted) ** 2))

    result = minimize_scalar(loss, bounds=bounds, method="bounded")
    return float(result.x)
