from __future__ import annotations

import hashlib

import httpx
import pytest
import respx

from athletevalue.schemas.source import LicenseTag
from athletevalue.sources.cache import ArtifactCache, ArtifactNotFoundError, OfflineError

URL = "https://example.org/data.parquet"


@respx.mock
def test_downloads_once_and_records_provenance(tmp_path):
    route = respx.get(URL).mock(return_value=httpx.Response(200, content=b"hello"))
    cache = ArtifactCache(tmp_path)
    first = cache.fetch(URL, relative_path="raw/data.parquet", license_tag=LicenseTag.MIT)
    second = cache.fetch(URL, relative_path="raw/data.parquet", license_tag=LicenseTag.MIT)
    assert route.call_count == 1
    assert first.sha256 == hashlib.sha256(b"hello").hexdigest() == second.sha256
    assert cache.manifest()["raw/data.parquet"]["license"] == "mit"
    reference = first.to_source_reference(kind="data_release", title="t")
    assert reference.is_verifiable


def test_offline_without_cache_raises(tmp_path):
    with pytest.raises(OfflineError):
        ArtifactCache(tmp_path, offline=True).fetch(
            URL, relative_path="x", license_tag=LicenseTag.MIT
        )


@respx.mock
def test_missing_upstream_raises(tmp_path):
    respx.get(URL).mock(return_value=httpx.Response(404))
    with pytest.raises(ArtifactNotFoundError):
        ArtifactCache(tmp_path).fetch(URL, relative_path="x", license_tag=LicenseTag.MIT)
    assert not (tmp_path / "x").exists()
