"""Equity in Athletics Data Analysis (EADA), U.S. Department of Education.

Public-domain institution and sport-level athletics finances, one archive per
academic year. The file list is fetched from the site's JSON API because file
names are not consistent across years (some use a space, some an underscore).
EADA ``Year`` is the ending year of the academic year, which matches this
package's season key.
"""

from __future__ import annotations

import json
import warnings
import zipfile

import polars as pl

from athletevalue.schemas.source import LicenseTag, SourceKind, SourceReference
from athletevalue.sources.cache import ArtifactCache, RawArtifact

API_BASE = "https://ope.ed.gov/athletics/api/dataFiles"

SCHOOL_COLUMNS = (
    "unitid",
    "institution_name",
    "state_cd",
    "classification_name",
    "Sports",
    "PARTIC_MEN",
    "REV_MEN",
    "EXP_MEN",
)


class EadaYearUnavailableError(LookupError):
    pass


class EadaClient:
    def __init__(self, cache: ArtifactCache) -> None:
        self.cache = cache

    def file_list(self, *, refresh: bool = False) -> list[dict[str, object]]:
        artifact = self.cache.fetch(
            f"{API_BASE}/fileList",
            relative_path="raw/eada/fileList.json",
            license_tag=LicenseTag.PUBLIC_DOMAIN,
            refresh=refresh,
        )
        entries: list[dict[str, object]] = json.loads(artifact.path.read_text(encoding="utf-8"))
        return entries

    def archive_name(self, season: int) -> str:
        for entry in self.file_list():
            name = str(entry.get("FileName", ""))
            if entry.get("Year") == season and "All" not in name and entry.get("Format"):
                return name
        raise EadaYearUnavailableError(f"EADA has no per-sport archive for {season}")

    def archive(self, season: int) -> RawArtifact:
        name = self.archive_name(season)
        return self.cache.fetch(
            f"{API_BASE}/file?fileName={name.replace(' ', '%20')}",
            relative_path=f"raw/eada/{name.replace(' ', '_')}",
            license_tag=LicenseTag.PUBLIC_DOMAIN,
        )

    def schools(self, season: int) -> pl.DataFrame:
        """Sport-level rows for *season*, restricted to the columns this package uses."""
        derived = f"derived/eada/schools_{season}.parquet"
        target = self.cache.path_for(derived)
        if target.exists():
            return pl.read_parquet(target)
        archive = self.archive(season)
        with zipfile.ZipFile(archive.path) as bundle:
            member = _school_member(bundle.namelist(), season)
            content = bundle.read(member)
        with warnings.catch_warnings():
            # polars' own calamine reader calls a from_arrow overload it has deprecated;
            # the warning is about polars internals, not this call.
            warnings.filterwarnings("ignore", message=r"from_arrow\(", category=FutureWarning)
            frame = pl.read_excel(content, columns=list(SCHOOL_COLUMNS), infer_schema_length=None)
        frame = frame.with_columns(
            pl.col("unitid").cast(pl.Int64),
            *(
                pl.col(c).cast(pl.Float64, strict=False)
                for c in ("PARTIC_MEN", "REV_MEN", "EXP_MEN")
            ),
            pl.lit(season).alias("season"),
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(target)
        self.cache.record_derived(derived, source=archive)
        return frame

    def reference(self, season: int) -> SourceReference:
        return self.archive(season).to_source_reference(
            kind=SourceKind.GOVERNMENT_DATA, title=f"EADA sport-level data, {season - 1}-{season}"
        )


def _school_member(names: list[str], season: int) -> str:
    candidates = [
        name
        for name in names
        if name.lower().startswith("schools") and name.lower().endswith((".xls", ".xlsx"))
    ]
    if not candidates:
        raise EadaYearUnavailableError(f"no Schools spreadsheet in EADA archive for {season}")
    return sorted(candidates, key=lambda name: name.lower().endswith(".xlsx"), reverse=True)[0]
