"""Loading and lookup for the assumption registry."""

from __future__ import annotations

import hashlib
import json
import tomllib
from collections.abc import Iterator
from functools import cache, cached_property
from importlib import resources
from pathlib import Path
from typing import Any, Self

from athletevalue.assumptions.models import Assumption

_DATA_PACKAGE = "athletevalue.assumptions.data"


class UnknownAssumptionError(KeyError):
    """Raised when a model asks for a parameter that is not in the registry."""


class AssumptionRegistry:
    def __init__(self, assumptions: dict[str, Assumption]) -> None:
        self._assumptions = assumptions

    @classmethod
    def load(cls, extra_paths: tuple[Path, ...] = ()) -> Self:
        """Load the packaged registry plus any user TOML files.

        A user file may override a packaged entry. Overrides are how a reader
        runs a what-if, so the override must restate its basis and evidence.
        """
        assumptions: dict[str, Assumption] = {}
        for path in _packaged_files():
            _ingest(assumptions, _read(path), str(path), allow_override=False)
        for path in extra_paths:
            _ingest(assumptions, _read(path), str(path), allow_override=True)
        return cls(assumptions)

    def get(self, assumption_id: str) -> Assumption:
        try:
            return self._assumptions[assumption_id]
        except KeyError:
            raise UnknownAssumptionError(
                f"{assumption_id!r} is not in the assumption registry. Add it with its "
                "evidence rather than hardcoding the value."
            ) from None

    def __iter__(self) -> Iterator[Assumption]:
        return iter(self._assumptions.values())

    def __len__(self) -> int:
        return len(self._assumptions)

    def __contains__(self, assumption_id: object) -> bool:
        return assumption_id in self._assumptions

    @cached_property
    def digest(self) -> str:
        """SHA-256 of every entry, so caches keyed on it change when any entry does."""
        payload = json.dumps(
            [self._assumptions[key].model_dump(mode="json") for key in sorted(self._assumptions)],
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()


@cache
def default_registry() -> AssumptionRegistry:
    return AssumptionRegistry.load()


def _read(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _packaged_files() -> Iterator[Path]:
    data_root = resources.files(_DATA_PACKAGE)
    for entry in sorted(data_root.iterdir(), key=lambda item: item.name):
        if entry.name.endswith(".toml"):
            with resources.as_file(entry) as path:
                yield path


def _ingest(
    target: dict[str, Assumption], document: dict[str, Any], origin: str, *, allow_override: bool
) -> None:
    for assumption_id, body in document.items():
        if not isinstance(body, dict):
            raise ValueError(f"{origin}: entry {assumption_id!r} must be a table")
        if assumption_id in target and not allow_override:
            raise ValueError(f"{origin}: duplicate assumption id {assumption_id!r}")
        fields = dict(body)
        if isinstance(fields.get("value"), list):
            fields["value"] = tuple(fields["value"])
        target[assumption_id] = Assumption(assumption_id=assumption_id, **fields)
