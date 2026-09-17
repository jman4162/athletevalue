from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime, timedelta

import httpx
import polars as pl
import pytest
import respx

from athletevalue.schemas.source import LicenseTag
from athletevalue.sources.cache import (
    ArtifactCache,
    ArtifactMismatchError,
    ArtifactNotFoundError,
    OfflineError,
    Pin,
    SourceUnavailableError,
    derived_path,
    load_pins,
)

URL = "https://example.org/data.parquet"
OTHER = "https://example.org/other.parquet"


def _pin(content: bytes) -> dict[str, Pin]:
    return {URL: Pin(hashlib.sha256(content).hexdigest(), len(content))}


def _cache(tmp_path, **kwargs) -> ArtifactCache:
    kwargs.setdefault("pins", {})
    kwargs.setdefault("sleep", lambda _seconds: None)
    return ArtifactCache(tmp_path, **kwargs)


def _fetch(cache: ArtifactCache, url: str = URL, **kwargs):
    return cache.fetch(url, relative_path="raw/data.parquet", license_tag=LicenseTag.MIT, **kwargs)


@respx.mock
def test_downloads_once_and_records_provenance(tmp_path):
    route = respx.get(URL).mock(return_value=httpx.Response(200, content=b"hello"))
    cache = _cache(tmp_path)
    first = _fetch(cache)
    second = _fetch(cache)
    assert route.call_count == 1
    assert first.sha256 == hashlib.sha256(b"hello").hexdigest() == second.sha256
    entry = cache.manifest()["raw/data.parquet"]
    assert entry["license"] == "mit"
    assert entry["kind"] == "raw"
    assert first.relative_path == "raw/data.parquet"
    assert not first.pinned
    reference = first.to_source_reference(kind="data_release", title="t")
    assert reference.is_verifiable


def test_offline_without_cache_raises(tmp_path):
    with pytest.raises(OfflineError):
        _fetch(_cache(tmp_path, offline=True))


@respx.mock
def test_missing_upstream_raises(tmp_path):
    respx.get(URL).mock(return_value=httpx.Response(404))
    with pytest.raises(ArtifactNotFoundError):
        _fetch(_cache(tmp_path))
    assert not (tmp_path / "raw/data.parquet").exists()


@respx.mock
def test_pinned_download_matches(tmp_path):
    respx.get(URL).mock(return_value=httpx.Response(200, content=b"hello"))
    assert _fetch(_cache(tmp_path, pins=_pin(b"hello"))).pinned


@respx.mock
def test_download_that_differs_from_its_pin_is_rejected_and_not_kept(tmp_path):
    respx.get(URL).mock(return_value=httpx.Response(200, content=b"changed"))
    with pytest.raises(ArtifactMismatchError) as error:
        _fetch(_cache(tmp_path, pins=_pin(b"hello")))
    assert hashlib.sha256(b"hello").hexdigest() in str(error.value)
    assert hashlib.sha256(b"changed").hexdigest() in str(error.value)
    assert not (tmp_path / "raw/data.parquet").exists()
    assert not (tmp_path / "raw/data.parquet.part").exists()


@respx.mock
def test_allow_unpinned_accepts_a_changed_upstream_file(tmp_path):
    respx.get(URL).mock(return_value=httpx.Response(200, content=b"changed"))
    artifact = _fetch(_cache(tmp_path, pins=_pin(b"hello"), allow_unpinned=True))
    assert artifact.path.read_bytes() == b"changed"
    assert not artifact.pinned


@respx.mock
def test_cached_copy_that_differs_from_its_pin_is_downloaded_again(tmp_path):
    route = respx.get(URL).mock(
        side_effect=[httpx.Response(200, content=b"old"), httpx.Response(200, content=b"hello")]
    )
    _fetch(_cache(tmp_path))
    artifact = _fetch(_cache(tmp_path, pins=_pin(b"hello")))
    assert route.call_count == 2
    assert artifact.pinned
    assert artifact.path.read_bytes() == b"hello"


@respx.mock
def test_offline_cached_copy_that_differs_from_its_pin_raises(tmp_path):
    respx.get(URL).mock(return_value=httpx.Response(200, content=b"old"))
    _fetch(_cache(tmp_path))
    with pytest.raises(ArtifactMismatchError):
        _fetch(_cache(tmp_path, pins=_pin(b"hello"), offline=True))


@respx.mock
def test_damaged_cached_file_is_downloaded_again(tmp_path):
    route = respx.get(URL).mock(return_value=httpx.Response(200, content=b"hello"))
    cache = _cache(tmp_path)
    artifact = _fetch(cache)
    artifact.path.write_bytes(b"hellp")
    assert _fetch(cache).path.read_bytes() == b"hello"
    assert route.call_count == 2
    assert [issue.problem for issue in cache.verify()] == []


@respx.mock
def test_server_errors_and_rate_limits_are_retried_with_backoff(tmp_path):
    waits: list[float] = []
    route = respx.get(URL).mock(
        side_effect=[
            httpx.Response(503),
            httpx.Response(429, headers={"Retry-After": "7"}),
            httpx.ConnectError("reset"),
            httpx.Response(200, content=b"hello"),
        ]
    )
    artifact = _fetch(_cache(tmp_path, sleep=waits.append))
    assert artifact.path.read_bytes() == b"hello"
    assert route.call_count == 4
    assert waits == [1.0, 7.0, 4.0]


@respx.mock
def test_retries_stop_after_the_last_attempt(tmp_path):
    respx.get(URL).mock(return_value=httpx.Response(502))
    with pytest.raises(SourceUnavailableError) as error:
        _fetch(_cache(tmp_path, max_attempts=2))
    assert error.value.status == 502


@respx.mock
def test_forbidden_is_not_retried(tmp_path):
    route = respx.get(URL).mock(return_value=httpx.Response(403))
    with pytest.raises(SourceUnavailableError):
        _fetch(_cache(tmp_path))
    assert route.call_count == 1


@respx.mock
def test_stale_copy_is_refreshed_and_kept_when_refresh_fails(tmp_path):
    route = respx.get(URL).mock(
        side_effect=[
            httpx.Response(200, content=b"v1"),
            httpx.Response(200, content=b"v2"),
            httpx.Response(403),
        ]
    )
    cache = _cache(tmp_path)
    _fetch(cache)
    assert _fetch(cache, max_age=timedelta(days=30)).path.read_bytes() == b"v1"
    _age_entry(cache, days=31)
    assert _fetch(cache, max_age=timedelta(days=30)).path.read_bytes() == b"v2"
    _age_entry(cache, days=31)
    assert _fetch(cache, max_age=timedelta(days=30)).path.read_bytes() == b"v2"
    assert route.call_count == 3


def _age_entry(cache: ArtifactCache, *, days: int) -> None:
    manifest = cache.manifest()
    manifest["raw/data.parquet"]["retrieved_at"] = (
        datetime.now(UTC) - timedelta(days=days)
    ).isoformat()
    cache.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


@respx.mock
def test_derived_frame_is_reused_until_an_input_changes(tmp_path):
    respx.get(URL).mock(
        side_effect=[httpx.Response(200, content=b"v1"), httpx.Response(200, content=b"v2")]
    )
    cache = _cache(tmp_path)
    source = _fetch(cache)
    relative = derived_path("test", "frame", {"k": 1})
    frame = pl.DataFrame({"x": [1, 2]})
    assert cache.read_derived(relative) is None
    cache.write_derived(relative, frame, sources=[source], settings={"k": 1})
    reused = cache.read_derived(relative)
    assert reused is not None and reused.equals(frame)
    entry = cache.manifest()[relative]
    assert entry["licenses"] == ["mit"]
    assert entry["sources"][0]["sha256"] == source.sha256

    _fetch(cache, refresh=True)
    assert cache.read_derived(relative) is None
    assert [issue.problem for issue in cache.verify()] == ["inputs changed since it was built"]


def test_derived_path_depends_on_settings_and_model_version(monkeypatch):
    first = derived_path("rapm", "baseline_2026", {"lambda": 3000})
    assert first == derived_path("rapm", "baseline_2026", {"lambda": 3000})
    assert first != derived_path("rapm", "baseline_2026", {"lambda": 1000})
    monkeypatch.setattr("athletevalue.sources.cache.MODEL_VERSION", "mbb-v9")
    assert first != derived_path("rapm", "baseline_2026", {"lambda": 3000})


def test_derived_file_edited_on_disk_is_not_reused(tmp_path):
    cache = _cache(tmp_path)
    relative = derived_path("test", "frame")
    cache.write_derived(relative, pl.DataFrame({"x": [1]}), sources=[])
    target = cache.path_for(relative)
    pl.DataFrame({"x": [2]}).write_parquet(target)
    os.utime(target, ns=(0, 0))
    assert cache.read_derived(relative) is None


@respx.mock
def test_summary_and_clear(tmp_path):
    respx.get(URL).mock(return_value=httpx.Response(200, content=b"hello"))
    cache = _cache(tmp_path, pins=_pin(b"hello"))
    source = _fetch(cache)
    cache.write_derived(derived_path("t", "f"), pl.DataFrame({"x": [1]}), sources=[source])
    summary = cache.summary()
    assert (summary.raw_files, summary.derived_files, summary.pinned, summary.unpinned) == (
        1,
        1,
        1,
        0,
    )
    assert cache.clear(derived_only=True) == 1
    assert list(cache.manifest()) == ["raw/data.parquet"]
    assert source.path.exists()
    assert cache.clear() == 1
    assert cache.manifest() == {}
    assert not source.path.exists()


def test_packaged_pins_are_well_formed():
    pins = load_pins()
    for url, pin in pins.items():
        assert url.startswith("https://")
        assert len(pin.sha256) == 64
        assert pin.byte_count > 0
