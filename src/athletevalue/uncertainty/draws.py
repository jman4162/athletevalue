"""Monte Carlo helpers shared by the wins, economics and market layers.

Uncertainty moves between layers as arrays of draws, not as summary intervals,
so products and differences of uncertain quantities keep their shape.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
from numpy.typing import NDArray

from athletevalue.schemas.estimate import Estimate
from athletevalue.schemas.evidence import EvidenceStatus

FloatArray = NDArray[np.float64]


def z_for_level(level: float) -> float:
    """Two-sided standard-normal quantile for a central interval at *level*."""
    return NormalDist().inv_cdf(0.5 + level / 2)


def normal_draws(mean: float, sd: float, n: int, rng: np.random.Generator) -> FloatArray:
    if sd <= 0:
        return np.full(n, mean, dtype=np.float64)
    return rng.normal(mean, sd, n)


def lognormal_from_range(
    low: float, high: float, level: float, n: int, rng: np.random.Generator
) -> FloatArray:
    """Draws whose central *level* interval is (low, high), log-symmetric about the median."""
    if low <= 0 or high < low:
        raise ValueError(f"range must be positive and ordered, got ({low}, {high})")
    median = math.sqrt(low * high)
    sigma = math.log(high / low) / (2 * z_for_level(level))
    return median * np.exp(rng.normal(0.0, 1.0, n) * sigma)


def uniform_between(low: float, high: float, n: int, rng: np.random.Generator) -> FloatArray:
    if high < low:
        raise ValueError(f"range must be ordered, got ({low}, {high})")
    return rng.uniform(low, high, n)


def summarize(
    values: FloatArray, *, unit: str, status: EvidenceStatus, method: str, level: float
) -> Estimate:
    """Median and central interval of *values*."""
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"{method}: draws must be non-empty and finite")
    tail = (1 - level) / 2
    lower, median, upper = np.quantile(values, [tail, 0.5, 1 - tail])
    return Estimate(
        value=float(median),
        lower=float(lower),
        upper=float(upper),
        level=level,
        unit=unit,
        status=status,
        method=method,
        se=float(np.std(values)),
    )
