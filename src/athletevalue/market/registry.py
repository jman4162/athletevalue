"""The community registry of publicly sourced deals."""

from __future__ import annotations

from importlib import resources

import polars as pl

from athletevalue.identity.names import normalize_name
from athletevalue.schemas.registry import DealRecord

_DATA_PACKAGE = "athletevalue.data.deal_registry"
REGISTRY_FILE = "mbb_deals.csv"


def load_deal_registry(frame: pl.DataFrame | None = None) -> list[DealRecord]:
    """Validate every row of the packaged registry (or *frame*) as a ``DealRecord``."""
    if frame is None:
        with resources.as_file(resources.files(_DATA_PACKAGE) / REGISTRY_FILE) as path:
            frame = pl.read_csv(path, infer_schema_length=0)
    records = []
    for row in frame.iter_rows(named=True):
        cleaned = {key: value for key, value in row.items() if value not in (None, "")}
        record = DealRecord.model_validate(cleaned)
        if record.usable:
            records.append(record)
    return records


def observed_annual_pay(
    records: list[DealRecord], *, name: str, school: str, season: int
) -> tuple[float, list[DealRecord]] | None:
    """Sum of annualized disclosed deals for a player-season, with the deals used."""
    wanted_name, wanted_school = normalize_name(name), normalize_name(school)
    matches = [
        r
        for r in records
        if r.season == season
        and normalize_name(r.athlete_name) == wanted_name
        and normalize_name(r.school) == wanted_school
    ]
    if not matches:
        return None
    return sum(r.annualized_value for r in matches), matches
