"""Bart Torvik team results, used only to check this package's team ratings.

Basis for the access pattern: https://barttorvik.com/robots.txt (read 2026-09-17)
disallows the interactive pages (``/db.php``, ``/box.php``, ``/results.php``,
``/playerstat.php``, ``/*.json`` and others) and sets ``Crawl-Delay: 10``; the
per-season ``{season}_team_results.csv`` files are not disallowed. The package
fetches one file per season and caches it, which stays within that delay. The
site refuses requests from some cloud address ranges with HTTP 403, so callers
treat this source as optional. Nothing from it is bundled or used to fit a
shipped value.
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
        """adjoe, adjde and their difference per team. Raises ``SourceUnavailableError`` on 403."""
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
