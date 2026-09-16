"""The curated map from stats.ncaa.org team names to IPEDS unit ids used by EADA."""

from __future__ import annotations

from functools import cache
from importlib import resources

import polars as pl

_DATA_PACKAGE = "athletevalue.data.identity"


@cache
def team_crosswalk() -> pl.DataFrame:
    """team, unitid, eada_institution_name. A team may map to several unit ids over time."""
    with resources.as_file(resources.files(_DATA_PACKAGE) / "team_crosswalk.csv") as path:
        return pl.read_csv(path, schema_overrides={"unitid": pl.Int64})
