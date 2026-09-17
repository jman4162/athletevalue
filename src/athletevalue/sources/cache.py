"""A content-addressed download cache with a provenance manifest.

Every artifact records its URL, SHA-256, size, retrieval time and license. The
manifest is what lets a result say which bytes it was computed from.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
import platformdirs

from athletevalue.schemas.source import LicenseTag, SourceKind, SourceReference
from athletevalue.versions import __version__

CACHE_ENV = "ATHLETEVALUE_CACHE_DIR"
USER_AGENT = f"athletevalue/{__version__} (+https://github.com/jman4162/athletevalue)"
_TIMEOUT_SECONDS = 300.0
_CHUNK_BYTES = 1 << 20


class OfflineError(RuntimeError):
    """Raised when an artifact is needed, is not cached, and downloads are disabled."""


class ArtifactNotFoundError(RuntimeError):
    """Raised when the upstream server reports that an artifact does not exist."""


class SourceUnavailableError(RuntimeError):
    """Raised when a source refuses or fails a download (for example a 403 to cloud runners)."""

    def __init__(self, url: str, status: int) -> None:
        super().__init__(f"{url} returned HTTP {status}")
        self.url = url
        self.status = status


@dataclass(frozen=True)
class RawArtifact:
    path: Path
    url: str
    sha256: str
    byte_count: int
    retrieved_at: datetime
    license: LicenseTag

    def to_source_reference(self, *, kind: SourceKind, title: str) -> SourceReference:
        return SourceReference(
            kind=kind,
            title=title,
            url=self.url,
            retrieved_at=self.retrieved_at,
            content_sha256=self.sha256,
            license=self.license,
        )


class ArtifactCache:
    """Downloads files once and remembers where they came from."""

    def __init__(
        self, root: Path, *, offline: bool = False, client: httpx.Client | None = None
    ) -> None:
        self.root = root
        self.offline = offline
        self._client = client

    @classmethod
    def default(cls, *, offline: bool = False) -> ArtifactCache:
        override = os.environ.get(CACHE_ENV)
        root = Path(override) if override else Path(platformdirs.user_cache_dir("athletevalue"))
        return cls(root, offline=offline)

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

    def fetch(
        self, url: str, *, relative_path: str, license_tag: LicenseTag, refresh: bool = False
    ) -> RawArtifact:
        """Return the cached artifact, downloading it first if needed."""
        target = self.path_for(relative_path)
        entry = self.manifest().get(relative_path)
        if target.exists() and entry is not None and not refresh:
            return _artifact_from_entry(target, entry)
        if self.offline:
            raise OfflineError(f"{relative_path} is not cached and downloads are disabled ({url})")
        return self._download(url, target, relative_path, license_tag)

    def _download(
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
                if response.is_error:
                    raise SourceUnavailableError(url, response.status_code)
                with partial.open("wb") as handle:
                    for chunk in response.iter_bytes(_CHUNK_BYTES):
                        handle.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
        finally:
            if self._client is None:
                client.close()
        partial.replace(target)
        artifact = RawArtifact(
            path=target,
            url=url,
            sha256=digest.hexdigest(),
            byte_count=size,
            retrieved_at=datetime.now(UTC),
            license=license_tag,
        )
        self._record(relative_path, artifact)
        return artifact

    def record_derived(self, relative_path: str, *, source: RawArtifact) -> None:
        """Register a file computed locally from *source*, inheriting its provenance."""
        path = self.path_for(relative_path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        derived = RawArtifact(
            path=path,
            url=source.url,
            sha256=digest,
            byte_count=path.stat().st_size,
            retrieved_at=source.retrieved_at,
            license=source.license,
        )
        self._record(relative_path, derived)

    def _record(self, relative_path: str, artifact: RawArtifact) -> None:
        manifest = self.manifest()
        manifest[relative_path] = {
            "url": artifact.url,
            "sha256": artifact.sha256,
            "bytes": artifact.byte_count,
            "retrieved_at": artifact.retrieved_at.isoformat(),
            "license": str(artifact.license),
        }
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.manifest_path.with_suffix(".json.part")
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(self.manifest_path)


def _artifact_from_entry(path: Path, entry: dict[str, object]) -> RawArtifact:
    return RawArtifact(
        path=path,
        url=str(entry["url"]),
        sha256=str(entry["sha256"]),
        byte_count=int(str(entry["bytes"])),
        retrieved_at=datetime.fromisoformat(str(entry["retrieved_at"])),
        license=LicenseTag(str(entry["license"])),
    )
