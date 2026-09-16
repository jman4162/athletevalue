"""Deterministic identifiers."""

from __future__ import annotations

import hashlib


def stable_id(prefix: str, *parts: object) -> str:
    """Return ``prefix:`` plus a short hash of *parts*.

    ``None`` and missing parts hash the same way, so an id does not change when an
    optional field is added as ``None``.
    """
    payload = "\x1f".join("" if part is None else str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"
