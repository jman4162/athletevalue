"""A download cache populated with synthetic seasons, so the CLI and sessions run offline.

Seasons default to 2041 and 2042: no pin exists for those URLs, so the shipped pins
never reject the synthetic bytes.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from athletevalue.assumptions.registry import default_registry
from athletevalue.sources.sportsdataverse import SdvClient, SdvDataset
from athletevalue.valuation.season import baseline_rapm
from tests.fixtures.raw_season import make_raw_season

SEASONS = (2041, 2042)


def populate_cache(root: Path, seasons: tuple[int, ...] = SEASONS) -> Path:
    """Write every SportsDataverse file the pipeline reads for *seasons* under *root*."""
    manifest: dict[str, dict[str, object]] = {}
    client = SdvClient.__new__(SdvClient)  # only url() is used, which needs no cache
    person_ids: dict[str, int] = {}
    for offset, season in enumerate(seasons):
        frames, _ = make_raw_season(seed=offset, season=season)
        reference = baseline_rapm(frames, default_registry()).table.filter(~pl.col("pooled"))
        for athlete in reference["athlete_id"].to_list():
            person_ids.setdefault(athlete, 10_000 + len(person_ids))
        tables = {
            SdvDataset.POSSESSIONS: frames.possessions,
            SdvDataset.TEAM_IDS: frames.team_ids,
            SdvDataset.SCHEDULE: frames.schedule,
            SdvDataset.ESPN_SCHEDULE: frames.espn_schedule,
            SdvDataset.PLAYER_BOX: frames.player_box,
            SdvDataset.REFERENCE_RAPM: reference.select(
                pl.col("athlete_id").alias("player_id"),
                pl.col("athlete_id")
                .replace_strict(person_ids, return_dtype=pl.Int64)
                .alias("person_id"),
                pl.col("net").alias("rapm_net"),
                "off_poss",
            ),
        }
        for dataset, table in tables.items():
            assert table is not None
            relative = f"raw/sportsdataverse/{dataset.value}/{dataset.file_stem}_{season}.parquet"
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            table.write_parquet(path)
            manifest[relative] = {
                "kind": "raw",
                "url": client.url(dataset, season),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
                "retrieved_at": datetime(2042, 4, 10, tzinfo=UTC).isoformat(),
                "license": str(dataset.license_tag),
                "mtime_ns": path.stat().st_mtime_ns,
            }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return root
