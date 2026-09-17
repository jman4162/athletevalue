"""The committed snapshot is anonymous, current in schema, and records passing gates."""

from __future__ import annotations

import json
from pathlib import Path

from athletevalue.valuation.snapshot import SCHEMA_VERSION, assert_anonymous

SNAPSHOT = Path(__file__).parents[2] / "docs" / "_static" / "data" / "snapshot.json"


def _load():
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))


def test_snapshot_has_no_identifying_keys():
    assert_anonymous(_load())


def test_snapshot_schema_and_gates():
    snapshot = _load()
    assert snapshot["schema_version"] == SCHEMA_VERSION
    for season, section in snapshot["seasons"].items():
        failed = [gate["gate"] for gate in section["gates"] if not gate["passed"]]
        assert not failed, (season, failed)


def test_snapshot_strings_are_short_labels_or_known_prose():
    """Every free-text string sits under a key whose content the package writes itself."""
    allowed_parents = {"detail", "notes", "note", "gate", "headline_definition", "role", "quadrant"}
    allowed_parents |= {"data_through", "model_version", "package_version"}

    def walk(value, parent):
        if isinstance(value, dict):
            for key, item in value.items():
                walk(item, key)
        elif isinstance(value, list):
            for item in value:
                walk(item, parent)
        elif isinstance(value, str):
            assert parent in allowed_parents, (parent, value)

    walk(_load(), None)
