"""Nothing in the modelling layer may bypass the assumption registry."""

from __future__ import annotations

from pathlib import Path

from athletevalue.assumptions.audit import scan_paths, scan_source

SRC = Path(__file__).resolve().parents[2] / "src" / "athletevalue"
MODELLING_PACKAGES = tuple(
    SRC / name for name in ("impact", "wins", "economics", "market", "valuation")
)


def test_scanner_flags_a_bare_float():
    assert [v.literal for v in scan_source("def f():\n    return 0.85\n")] == ["0.85"]


def test_scanner_allows_structural_constants_and_docstrings():
    source = 'def f(a, b):\n    """Uses 42 in prose."""\n    return (a + b) / 2 * 100\n'
    assert scan_source(source) == []


def test_modelling_layer_has_no_uncited_constants():
    violations = scan_paths(MODELLING_PACKAGES)
    assert not violations, "\n".join(str(v) for v in violations)
