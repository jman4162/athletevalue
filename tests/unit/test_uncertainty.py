from __future__ import annotations

import numpy as np
import pytest

from athletevalue.schemas.evidence import EvidenceStatus
from athletevalue.uncertainty.draws import lognormal_from_range, summarize


def test_lognormal_range_is_the_central_interval():
    draws = lognormal_from_range(2e6, 5e6, 0.8, 200_000, np.random.default_rng(0))
    low, median, high = np.quantile(draws, [0.1, 0.5, 0.9])
    assert low == pytest.approx(2e6, rel=0.01)
    assert high == pytest.approx(5e6, rel=0.01)
    assert median == pytest.approx(np.sqrt(10e12), rel=0.01)


def test_summarize_reports_median_and_quantiles():
    est = summarize(
        np.arange(101, dtype=float), unit="x", status=EvidenceStatus.DERIVED, method="m", level=0.8
    )
    assert (est.lower, est.value, est.upper) == pytest.approx((10.0, 50.0, 90.0))
    with pytest.raises(ValueError):
        summarize(
            np.array([np.nan]), unit="x", status=EvidenceStatus.DERIVED, method="m", level=0.8
        )
