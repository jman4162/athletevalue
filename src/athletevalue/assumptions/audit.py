"""Static check for uncited constants in the modelling layer.

Walks the AST of modelling modules and reports numeric literals that should be
registry entries. Identity elements, halving and the per-100 scale are exempt;
anything else is a modelling choice and needs evidence.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

ALLOWED_NUMBERS: frozenset[float] = frozenset({0, 1, 2, -1, 100, 0.5})


@dataclass(frozen=True)
class LiteralViolation:
    path: Path
    line: int
    literal: str

    def __str__(self) -> str:
        return (
            f"{self.path}:{self.line}: uncited constant {self.literal}. "
            "Move it to the assumption registry with its evidence."
        )


def scan_source(source: str, path: Path = Path("<memory>")) -> list[LiteralViolation]:
    return list(_walk(ast.parse(source), path))


def scan_paths(roots: tuple[Path, ...]) -> list[LiteralViolation]:
    violations: list[LiteralViolation] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            violations.extend(scan_source(path.read_text(encoding="utf-8"), path))
    return violations


def _walk(tree: ast.AST, path: Path) -> Iterator[LiteralViolation]:
    docstrings = _docstring_nodes(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or id(node) in docstrings:
            continue
        value = node.value
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        if float(value) in ALLOWED_NUMBERS:
            continue
        yield LiteralViolation(path=path, line=node.lineno, literal=repr(value))


def _docstring_nodes(tree: ast.AST) -> set[int]:
    found: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                found.add(id(body[0].value))
    return found
