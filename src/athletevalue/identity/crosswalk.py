"""The curated map from stats.ncaa.org team names to IPEDS unit ids used by EADA."""

from __future__ import annotations

from functools import cache
from importlib import resources

import polars as pl

_DATA_PACKAGE = "athletevalue.data.identity"


@cache
@cache
def team_aliases() -> dict[str, tuple[str, ...]]:
    """Other names for each stats.ncaa.org team: its institution names in the crosswalk.

    Box-score names abbreviate ("UNI", "SFA", "FDU") where ESPN spells the school out,
    so name matching also tries the institution name. ``team_name_aliases.csv`` adds the
    few names neither spelling contains, such as ESPN's "USC".
    """
    with resources.as_file(resources.files(_DATA_PACKAGE) / "team_name_aliases.csv") as path:
        curated = pl.read_csv(path).select("team", pl.col("alias").alias("name"))
    frame = (
        pl.concat(
            [
                team_crosswalk().select("team", pl.col("eada_institution_name").alias("name")),
                curated,
            ]
        )
        .drop_nulls()
        .group_by("team")
        .agg(pl.col("name").unique())
    )
    return {
        str(team): tuple(sorted(str(name) for name in names)) for team, names in frame.iter_rows()
    }


@cache
def team_crosswalk() -> pl.DataFrame:
    """team, unitid, eada_institution_name. A team may map to several unit ids over time."""
    with resources.as_file(resources.files(_DATA_PACKAGE) / "team_crosswalk.csv") as path:
        return pl.read_csv(path, schema_overrides={"unitid": pl.Int64})
