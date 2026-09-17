"""Figures render from the committed snapshot and write identical SVG bytes each time."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from athletevalue.viz import FIGURES, save_svg

SNAPSHOT = Path(__file__).parents[2] / "docs" / "_static" / "data" / "snapshot.json"


@pytest.fixture(scope="module")
def snapshot():
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))


@pytest.mark.parametrize("stem", sorted(FIGURES))
def test_figure_renders_to_stable_svg(stem, snapshot, tmp_path):
    first, second = tmp_path / "a.svg", tmp_path / "b.svg"
    save_svg(FIGURES[stem](snapshot), first)
    save_svg(FIGURES[stem](snapshot), second)
    content = first.read_bytes()
    assert content == second.read_bytes()
    assert b"<dc:date>" not in content
    assert len(content) < 400_000
