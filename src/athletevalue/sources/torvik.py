"""Bart Torvik team results, used only to check this package's team ratings.

The owner publishes season CSVs for bulk use and asks that pages not be scraped.
Nothing from this source is bundled or used to fit a shipped value.
"""

from __future__ import annotations

import polars as pl

from athletevalue.schemas.source import LicenseTag
from athletevalue.sources.cache import ArtifactCache

TEAM_RESULTS_URL = "https://barttorvik.com/{season}_team_results.csv"


class TorvikClient:
    def __init__(self, cache: ArtifactCache) -> None:
        self.cache = cache

    def team_results(self, season: int) -> pl.DataFrame:
        artifact = self.cache.fetch(
            TEAM_RESULTS_URL.format(season=season),
            relative_path=f"raw/torvik/{season}_team_results.csv",
            license_tag=LicenseTag.VALIDATION_ONLY,
        )
        frame = pl.read_csv(artifact.path, infer_schema_length=None)
        return frame.select(
            pl.col("team").cast(pl.Utf8),
            pl.col("conf").cast(pl.Utf8).alias("conference"),
            pl.col("adjoe").cast(pl.Float64),
            pl.col("adjde").cast(pl.Float64),
        ).with_columns((pl.col("adjoe") - pl.col("adjde")).alias("adj_em"))
