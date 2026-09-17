"""EADA parsing against invented workbooks in the two formats the archives use."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import httpx
import pytest
import respx

from athletevalue.sources.cache import ArtifactCache
from athletevalue.sources.eada import API_BASE, EadaClient, EadaYearUnavailableError

FIXTURES = Path(__file__).parents[1] / "fixtures" / "eada"
FILE_LIST = [
    {"FileName": "EADA 2010-2011.zip", "Year": 2011, "Format": "Excel"},
    {"FileName": "EADA_2024-2025.zip", "Year": 2025, "Format": "Excel"},
    {"FileName": "EADA All Years.zip", "Year": 2025, "Format": "Excel"},
    {"FileName": "EADA 2025-2026.zip", "Year": 2026, "Format": ""},
]


def _archive(workbook: Path) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        bundle.write(workbook, workbook.name)
        bundle.writestr("Schools.sav", b"not a spreadsheet")
        bundle.writestr("instLevel" + workbook.suffix, b"institution-level file, not read")
    return buffer.getvalue()


@pytest.fixture
def client(tmp_path):
    respx.get(f"{API_BASE}/fileList").mock(return_value=httpx.Response(200, json=FILE_LIST))
    respx.get(f"{API_BASE}/file?fileName=EADA%202010-2011.zip").mock(
        return_value=httpx.Response(200, content=_archive(FIXTURES / "Schools.xls"))
    )
    respx.get(f"{API_BASE}/file?fileName=EADA_2024-2025.zip").mock(
        return_value=httpx.Response(200, content=_archive(FIXTURES / "schools.xlsx"))
    )
    return EadaClient(ArtifactCache(tmp_path, pins={}))


@respx.mock
@pytest.mark.parametrize("season", [2011, 2025])
def test_schools_reads_both_workbook_formats(client, season):
    frame = client.schools(season)
    assert frame.columns[:3] == ["unitid", "institution_name", "state_cd"]
    assert "ClassificationOther" not in frame.columns
    assert frame["season"].unique().to_list() == [season]
    basketball = frame.filter(frame["Sports"] == "Basketball").sort("unitid")
    assert basketball["REV_MEN"].to_list() == [12_500_000.0, None]
    assert basketball["unitid"].dtype.is_integer()


@respx.mock
def test_parsed_schools_are_cached_as_a_derived_file(client):
    first = client.schools(2025)
    offline = EadaClient(ArtifactCache(client.cache.root, offline=True, pins={}))
    assert offline.schools(2025).equals(first)
    derived = [path for path in offline.cache.manifest() if path.startswith("derived/eada/")]
    assert len(derived) == 1
    entry = offline.cache.manifest()[derived[0]]
    assert entry["licenses"] == ["public_domain"]


@respx.mock
def test_year_without_a_per_sport_archive(client):
    with pytest.raises(EadaYearUnavailableError):
        client.schools(2026)
    with pytest.raises(EadaYearUnavailableError):
        client.schools(2019)


@respx.mock
def test_archive_without_a_schools_workbook(tmp_path):
    respx.get(f"{API_BASE}/fileList").mock(return_value=httpx.Response(200, json=FILE_LIST))
    buffer = tmp_path / "empty.zip"
    with zipfile.ZipFile(buffer, "w") as bundle:
        bundle.writestr("readme.txt", b"")
    respx.get(f"{API_BASE}/file?fileName=EADA%202010-2011.zip").mock(
        return_value=httpx.Response(200, content=buffer.read_bytes())
    )
    with pytest.raises(EadaYearUnavailableError, match="no Schools spreadsheet"):
        EadaClient(ArtifactCache(tmp_path / "cache", pins={})).schools(2011)
