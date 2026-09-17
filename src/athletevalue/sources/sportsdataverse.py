"""SportsDataverse release assets for NCAA men's basketball.

The assets are parquet files attached to GitHub releases of an MIT-licensed
repository. No key or authentication is required. Season keys are the ending
year: 2026 is the 2025-26 season.
"""

from __future__ import annotations

from enum import StrEnum

import polars as pl

from athletevalue.schemas.source import LicenseTag, SourceKind, SourceReference
from athletevalue.sources.cache import ArtifactCache, ArtifactNotFoundError, RawArtifact

RELEASE_BASE = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"
FIRST_POSSESSION_SEASON = 2011
"""The source excludes 2010 because its substitution records are too sparse."""

LATEST_SEASON = 2026
"""Most recent completed season with published assets, as of this release."""


class SeasonUnavailableError(LookupError):
    """Raised when a season has no published asset."""


class SdvDataset(StrEnum):
    POSSESSIONS = "ncaa_mbb_possessions"
    TEAM_IDS = "ncaa_mbb_team_ids"
    SCHEDULE = "ncaa_mbb_schedule"
    PLAYER_BOX = "ncaa_mbb_player_box"
    REFERENCE_RAPM = "ncaa_mbb_rapm"
    ESPN_SCHEDULE = "espn_mens_college_basketball_schedules"

    @property
    def file_stem(self) -> str:
        if self is SdvDataset.ESPN_SCHEDULE:
            return "mbb_schedule"
        return self.value

    @property
    def label(self) -> str:
        return f"SportsDataverse release {self.value}"


class SdvClient:
    def __init__(self, cache: ArtifactCache) -> None:
        self.cache = cache

    def url(self, dataset: SdvDataset, season: int) -> str:
        return f"{RELEASE_BASE}/{dataset.value}/{dataset.file_stem}_{season}.parquet"

    def artifact(self, dataset: SdvDataset, season: int, *, refresh: bool = False) -> RawArtifact:
        if dataset is not SdvDataset.ESPN_SCHEDULE and season < FIRST_POSSESSION_SEASON:
            raise SeasonUnavailableError(
                f"{dataset.value} starts in {FIRST_POSSESSION_SEASON}; got {season}"
            )
        relative = f"raw/sportsdataverse/{dataset.value}/{dataset.file_stem}_{season}.parquet"
        try:
            return self.cache.fetch(
                self.url(dataset, season),
                relative_path=relative,
                license_tag=LicenseTag.MIT,
                refresh=refresh,
            )
        except ArtifactNotFoundError as error:
            raise SeasonUnavailableError(f"{dataset.value} has no asset for {season}") from error

    def frame(self, dataset: SdvDataset, season: int) -> pl.DataFrame:
        return pl.read_parquet(self.artifact(dataset, season).path)

    def reference(self, dataset: SdvDataset, season: int) -> SourceReference:
        return self.artifact(dataset, season).to_source_reference(
            kind=SourceKind.DATA_RELEASE, title=f"{dataset.label} ({season})"
        )
