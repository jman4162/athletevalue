"""Integer group codes that do not depend on polars' process-wide categories."""

from __future__ import annotations

import numpy as np
import polars as pl
from numpy.typing import NDArray


def first_seen_codes(values: pl.Series) -> NDArray[np.int64]:
    """Codes ``0..k-1`` numbered in order of first appearance.

    ``cast(pl.Categorical).to_physical()`` is not a substitute: in recent polars the
    categories are shared across the process, so codes depend on every string cast
    earlier. They are then neither contiguous nor stable between runs, and code that
    sizes arrays by ``codes.max() + 1`` gets empty groups.
    """
    array = values.to_numpy()
    _, first, inverse = np.unique(array, return_index=True, return_inverse=True)
    rank = np.empty(first.size, dtype=np.int64)
    rank[np.argsort(first, kind="stable")] = np.arange(first.size, dtype=np.int64)
    codes: NDArray[np.int64] = rank[inverse.reshape(-1)]
    return codes
