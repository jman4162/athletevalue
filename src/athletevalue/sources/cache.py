"""A download cache with a provenance manifest and pinned upstream hashes.

Every raw artifact records its URL, SHA-256, size, retrieval time and license.
Every derived file (a parquet computed locally from raw artifacts) records the
model version and settings that produced it and the digest of each input, so a
change to any of them makes the derived file stale instead of silently reused.

Upstream files behind the published numbers are pinned in
``athletevalue/data/pinned_artifacts.json``. A download or cached copy whose
digest differs from its pin raises ``ArtifactMismatchError`` unless the cache
was opened with ``allow_unpinned=True``. URLs without a pin (a season published
after this release, for example) are accepted and marked unpinned.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from importlib import resources
from pathlib import Path

import httpx
import platformdirs
import polars as pl

from athletevalue.schemas.source import LicenseTag, SourceKind, SourceReference
from athletevalue.versions import MODEL_VERSION, __version__

CACHE_ENV = "ATHLETEVALUE_CACHE_DIR"
USER_AGENT = f"athletevalue/{__version__} (+https://github.com/jman4162/athletevalue)"
PINS_FILE = "pinned_artifacts.json"
_DATA_PACKAGE = "athletevalue.data"
_TIMEOUT_SECONDS = 300.0
_CHUNK_BYTES = 1 << 20
_MAX_ATTEMPTS = 4
_MAX_BACKOFF_SECONDS = 60.0


class OfflineError(RuntimeError):
    """Raised when an artifact is needed, is not cached, and downloads are disabled."""


class ArtifactNotFoundError(RuntimeError):
    """Raised when the upstream server reports that an artifact does not exist."""


class SourceUnavailableError(RuntimeError):
    """Raised when a source refuses or fails a download (for example a 403 to cloud runners)."""

    def __init__(self, url: str, status: int | None, detail: str = "") -> None:
        reason = f"HTTP {status}" if status is not None else detail or "no response"
        super().__init__(f"{url} returned {reason}")
        self.url = url
        self.status = status


class ArtifactMismatchError(RuntimeError):
    """Raised when an artifact's bytes differ from the digest pinned for its URL."""

    def __init__(self, url: str, *, expected: str, actual: str) -> None:
        super().__init__(
            f"{url} has SHA-256 {actual}, but this release pins {expected}. The upstream file "
            "changed after the published numbers were computed, or the cached copy is damaged. "
            "Pass allow_unpinned=True (CLI: --allow-unpinned) to use it anyway; results may "
            "then differ from the documentation."
        )
        self.url = url
        self.expected = expected
        self.actual = actual


@dataclass(frozen=True)
class Pin:
    sha256: str
    byte_count: int


def load_pins() -> dict[str, Pin]:
    """Pinned digests shipped with the package, keyed by URL."""
    resource = resources.files(_DATA_PACKAGE) / PINS_FILE
    if not resource.is_file():
        return {}
    payload = json.loads(resource.read_text(encoding="utf-8"))
    return {
        url: Pin(sha256=str(item["sha256"]), byte_count=int(item["bytes"]))
        for url, item in payload["artifacts"].items()
    }


@dataclass(frozen=True)
class RawArtifact:
    path: Path
    url: str
    sha256: str
    byte_count: int
    retrieved_at: datetime
    license: LicenseTag
    relative_path: str = ""
    pinned: bool = False
    """True when a pin exists for the URL and the bytes match it."""

    def to_source_reference(self, *, kind: SourceKind, title: str) -> SourceReference:
        return SourceReference(
            kind=kind,
            title=title,
            url=self.url,
            retrieved_at=self.retrieved_at,
            content_sha256=self.sha256,
            license=self.license,
        )


@dataclass(frozen=True)
class CacheIssue:
    relative_path: str
    problem: str


@dataclass(frozen=True)
class CacheSummary:
    root: Path
    raw_files: int
    raw_bytes: int
    derived_files: int
    derived_bytes: int
    pinned: int
    unpinned: int


def derived_path(kind: str, name: str, settings: Mapping[str, object] | None = None) -> str:
    """Relative path for a derived parquet, keyed on the model version and *settings*.

    Two releases, or two settings, never share a derived file.
    """
    payload = json.dumps(
        {"model_version": MODEL_VERSION, "settings": dict(settings or {})},
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha256(payload.encode()).hexdigest()[:12]
    return f"derived/{kind}/{name}_{digest}.parquet"


class _RetryableError(Exception):
    def __init__(self, final: Exception, delay: float | None = None) -> None:
        super().__init__(str(final))
        self.final = final
        self.delay = delay


class ArtifactCache:
    """Downloads files once, checks them against pins, and remembers where they came from."""

    def __init__(
        self,
        root: Path,
        *,
        offline: bool = False,
        client: httpx.Client | None = None,
        pins: Mapping[str, Pin] | None = None,
        allow_unpinned: bool = False,
        max_attempts: int = _MAX_ATTEMPTS,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.root = root
        self.offline = offline
        self.allow_unpinned = allow_unpinned
        self.pins: Mapping[str, Pin] = load_pins() if pins is None else pins
        self._client = client
        self._max_attempts = max(1, max_attempts)
        self._sleep = sleep

    @classmethod
    def default(cls, *, offline: bool = False, allow_unpinned: bool = False) -> ArtifactCache:
        override = os.environ.get(CACHE_ENV)
        root = Path(override) if override else Path(platformdirs.user_cache_dir("athletevalue"))
        return cls(root, offline=offline, allow_unpinned=allow_unpinned)

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def path_for(self, relative_path: str) -> Path:
        return self.root / relative_path

    def manifest(self) -> dict[str, dict[str, object]]:
        if not self.manifest_path.exists():
            return {}
        data: dict[str, dict[str, object]] = json.loads(
            self.manifest_path.read_text(encoding="utf-8")
        )
        return data

    # Raw artifacts -------------------------------------------------------------

    def fetch(
        self,
        url: str,
        *,
        relative_path: str,
        license_tag: LicenseTag,
        refresh: bool = False,
        max_age: timedelta | None = None,
    ) -> RawArtifact:
        """Return the cached artifact, downloading it first if needed.

        A cached copy older than *max_age* is downloaded again when online; if that
        download fails, the older copy is returned.
        """
        target = self.path_for(relative_path)
        entry = self.manifest().get(relative_path)
        cached = None
        if target.exists() and entry is not None and str(entry.get("url")) == url:
            cached = self._checked_entry(relative_path, target, entry)
        if cached is not None and not refresh:
            stale = max_age is not None and datetime.now(UTC) - cached.retrieved_at > max_age
            if not stale or self.offline:
                return cached
            try:
                return self._download(url, target, relative_path, license_tag)
            except (SourceUnavailableError, ArtifactNotFoundError):
                return cached
        if self.offline:
            raise OfflineError(f"{relative_path} is not cached and downloads are disabled ({url})")
        return self._download(url, target, relative_path, license_tag)

    def _checked_entry(
        self, relative_path: str, target: Path, entry: dict[str, object]
    ) -> RawArtifact | None:
        """The cached artifact, or ``None`` when the file no longer matches its record.

        Size and modification time stand in for the digest; the file is hashed again
        only when either changed (or was never recorded).
        """
        recorded = str(entry["sha256"])
        if not _unchanged(target, entry):
            if _sha256_file(target) != recorded:
                return None
            self._update_entry(relative_path, mtime_ns=target.stat().st_mtime_ns)
        url = str(entry["url"])
        pin = self.pins.get(url)
        if pin is not None and pin.sha256 != recorded and not self.allow_unpinned:
            if self.offline:
                raise ArtifactMismatchError(url, expected=pin.sha256, actual=recorded)
            return None
        return _artifact_from_entry(target, relative_path, entry, pinned=_matches(pin, recorded))

    def _download(
        self, url: str, target: Path, relative_path: str, license_tag: LicenseTag
    ) -> RawArtifact:
        for attempt in range(1, self._max_attempts + 1):
            try:
                return self._download_once(url, target, relative_path, license_tag)
            except _RetryableError as error:
                if attempt == self._max_attempts:
                    raise error.final from error
                backoff = min(2.0 ** (attempt - 1), _MAX_BACKOFF_SECONDS)
                self._sleep(min(error.delay, _MAX_BACKOFF_SECONDS) if error.delay else backoff)
        raise AssertionError("unreachable")  # pragma: no cover

    def _download_once(
        self, url: str, target: Path, relative_path: str, license_tag: LicenseTag
    ) -> RawArtifact:
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        digest = hashlib.sha256()
        size = 0
        client = self._client or httpx.Client(
            follow_redirects=True, timeout=_TIMEOUT_SECONDS, headers={"User-Agent": USER_AGENT}
        )
        try:
            with client.stream("GET", url) as response:
                if response.status_code == httpx.codes.NOT_FOUND:
                    raise ArtifactNotFoundError(f"{url} returned 404")
                if response.status_code == httpx.codes.TOO_MANY_REQUESTS or (
                    response.status_code >= 500
                ):
                    raise _RetryableError(
                        SourceUnavailableError(url, response.status_code),
                        delay=_retry_after(response),
                    )
                if response.is_error:
                    raise SourceUnavailableError(url, response.status_code)
                with partial.open("wb") as handle:
                    for chunk in response.iter_bytes(_CHUNK_BYTES):
                        handle.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
        except httpx.TransportError as error:
            partial.unlink(missing_ok=True)
            raise _RetryableError(
                SourceUnavailableError(url, None, type(error).__name__)
            ) from error
        finally:
            if self._client is None:
                client.close()
        actual = digest.hexdigest()
        pin = self.pins.get(url)
        if pin is not None and pin.sha256 != actual and not self.allow_unpinned:
            partial.unlink(missing_ok=True)
            raise ArtifactMismatchError(url, expected=pin.sha256, actual=actual)
        partial.replace(target)
        artifact = RawArtifact(
            path=target,
            url=url,
            sha256=actual,
            byte_count=size,
            retrieved_at=datetime.now(UTC),
            license=license_tag,
            relative_path=relative_path,
            pinned=_matches(pin, actual),
        )
        self._record(
            relative_path,
            {
                "kind": "raw",
                "url": url,
                "sha256": actual,
                "bytes": size,
                "retrieved_at": artifact.retrieved_at.isoformat(),
                "license": str(license_tag),
                "mtime_ns": target.stat().st_mtime_ns,
            },
        )
        return artifact

    # Derived files -------------------------------------------------------------

    def read_derived(self, relative_path: str) -> pl.DataFrame | None:
        """The derived frame, or ``None`` when absent, altered, or built from other inputs."""
        target = self.path_for(relative_path)
        manifest = self.manifest()
        entry = manifest.get(relative_path)
        if not target.exists() or entry is None or entry.get("kind") != "derived":
            return None
        if entry.get("model_version") != MODEL_VERSION:
            return None
        if not _unchanged(target, entry) and _sha256_file(target) != entry["sha256"]:
            return None
        if _stale_sources(entry, manifest):
            return None
        return pl.read_parquet(target)

    def write_derived(
        self,
        relative_path: str,
        frame: pl.DataFrame,
        *,
        sources: Sequence[RawArtifact],
        settings: Mapping[str, object] | None = None,
    ) -> None:
        """Write *frame* and record the inputs, model version and settings behind it."""
        target = self.path_for(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        frame.write_parquet(partial)
        partial.replace(target)
        self._record(
            relative_path,
            {
                "kind": "derived",
                "sha256": _sha256_file(target),
                "bytes": target.stat().st_size,
                "mtime_ns": target.stat().st_mtime_ns,
                "created_at": datetime.now(UTC).isoformat(),
                "model_version": MODEL_VERSION,
                "settings": json.loads(json.dumps(dict(settings or {}), default=str)),
                "licenses": sorted({str(source.license) for source in sources}),
                "sources": [
                    {"path": source.relative_path, "url": source.url, "sha256": source.sha256}
                    for source in sources
                ],
            },
        )

    # Maintenance ---------------------------------------------------------------

    def summary(self) -> CacheSummary:
        raw_files = raw_bytes = derived_files = derived_bytes = pinned = unpinned = 0
        for relative_path, entry in self.manifest().items():
            size = int(str(entry.get("bytes", 0)))
            if _is_derived(relative_path, entry):
                derived_files += 1
                derived_bytes += size
                continue
            raw_files += 1
            raw_bytes += size
            if _matches(self.pins.get(str(entry.get("url"))), str(entry.get("sha256"))):
                pinned += 1
            else:
                unpinned += 1
        return CacheSummary(
            self.root, raw_files, raw_bytes, derived_files, derived_bytes, pinned, unpinned
        )

    def verify(self) -> list[CacheIssue]:
        """Hash every recorded file and report what no longer matches."""
        issues = []
        manifest = self.manifest()
        for relative_path, entry in sorted(manifest.items()):
            target = self.path_for(relative_path)
            if not target.exists():
                issues.append(CacheIssue(relative_path, "recorded but missing"))
                continue
            actual = _sha256_file(target)
            if actual != entry.get("sha256"):
                issues.append(CacheIssue(relative_path, "contents differ from the manifest"))
                continue
            if _is_derived(relative_path, entry):
                if entry.get("model_version") != MODEL_VERSION:
                    built_by = entry.get("model_version") or "an earlier release"
                    issues.append(CacheIssue(relative_path, f"built by {built_by}"))
                elif _stale_sources(entry, manifest):
                    issues.append(CacheIssue(relative_path, "inputs changed since it was built"))
                continue
            pin = self.pins.get(str(entry.get("url")))
            if pin is not None and pin.sha256 != actual:
                issues.append(CacheIssue(relative_path, f"differs from pinned {pin.sha256[:12]}"))
        return issues

    def clear(self, *, derived_only: bool = False) -> int:
        """Delete cached files (only derived ones if asked). Returns the number removed."""
        manifest = self.manifest()
        removed = [
            path for path, entry in manifest.items() if not derived_only or _is_derived(path, entry)
        ]
        if derived_only:
            shutil.rmtree(self.root / "derived", ignore_errors=True)
            self._write_manifest({p: e for p, e in manifest.items() if p not in set(removed)})
        else:
            for child in ("raw", "derived"):
                shutil.rmtree(self.root / child, ignore_errors=True)
            self.manifest_path.unlink(missing_ok=True)
        return len(removed)

    # Manifest ------------------------------------------------------------------

    def _record(self, relative_path: str, entry: dict[str, object]) -> None:
        manifest = self.manifest()
        manifest[relative_path] = entry
        self._write_manifest(manifest)

    def _update_entry(self, relative_path: str, **fields: object) -> None:
        manifest = self.manifest()
        if relative_path in manifest:
            manifest[relative_path].update(fields)
            self._write_manifest(manifest)

    def _write_manifest(self, manifest: dict[str, dict[str, object]]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.manifest_path.with_suffix(".json.part")
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(self.manifest_path)


def _unchanged(target: Path, entry: dict[str, object]) -> bool:
    """Size and modification time match the record, so the digest need not be recomputed."""
    stat = target.stat()
    return stat.st_size == int(str(entry["bytes"])) and entry.get("mtime_ns") == stat.st_mtime_ns


def _is_derived(relative_path: str, entry: dict[str, object]) -> bool:
    # Releases before 0.4.0 recorded derived files without a kind.
    return entry.get("kind") == "derived" or relative_path.startswith("derived/")


def _matches(pin: Pin | None, sha256: str) -> bool:
    return pin is not None and pin.sha256 == sha256


def _stale_sources(entry: dict[str, object], manifest: dict[str, dict[str, object]]) -> bool:
    sources = entry.get("sources")
    if not isinstance(sources, list):
        return True
    for source in sources:
        current = manifest.get(str(source.get("path")))
        if current is None or current.get("sha256") != source.get("sha256"):
            return True
    return False


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After", "")
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_from_entry(
    path: Path, relative_path: str, entry: dict[str, object], *, pinned: bool
) -> RawArtifact:
    return RawArtifact(
        path=path,
        url=str(entry["url"]),
        sha256=str(entry["sha256"]),
        byte_count=int(str(entry["bytes"])),
        retrieved_at=datetime.fromisoformat(str(entry["retrieved_at"])),
        license=LicenseTag(str(entry["license"])),
        relative_path=relative_path,
        pinned=pinned,
    )
