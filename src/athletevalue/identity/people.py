"""Linking a player across seasons.

stats.ncaa.org player ids (``athlete_id`` here) are issued per season, so the same
person has a new id each year. The SportsDataverse reference RAPM release carries a
``person_id`` that stays with the person across seasons and teams; this module maps
one to the other. A person can hold two athlete ids in one season (for example after
a mid-season roster change), so ``person_id`` is not unique within a season.
"""

from __future__ import annotations

import polars as pl

PEOPLE_SCHEMA = {"athlete_id": pl.Utf8, "person_id": pl.Utf8}


def person_map(reference_rapm: pl.DataFrame) -> pl.DataFrame:
    """athlete_id, person_id; one row per athlete id with a known person."""
    return (
        reference_rapm.select(
            pl.col("player_id").cast(pl.Utf8).alias("athlete_id"),
            pl.col("person_id").cast(pl.Utf8),
        )
        .drop_nulls()
        .unique(subset=["athlete_id"], keep="first", maintain_order=True)
    )


def with_person_ids(table: pl.DataFrame, people: pl.DataFrame | None) -> pl.DataFrame:
    """*table* with a ``person_id`` column, null where the person is unknown."""
    if "person_id" in table.columns:
        table = table.drop("person_id")
    if people is None:
        return table.with_columns(pl.lit(None, dtype=pl.Utf8).alias("person_id"))
    return table.join(people, on="athlete_id", how="left", maintain_order="left")
