"""Render the README figures from the committed snapshot.

Offline and deterministic: the same snapshot and matplotlib minor version give the
same SVG bytes, which CI checks with ``git diff --exit-code docs/_static``.

    uv run --extra viz python scripts/generate_readme_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from athletevalue.viz import FIGURES, save_svg

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "docs" / "_static"
SNAPSHOT = STATIC / "data" / "snapshot.json"


def main() -> None:
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    for stem, draw in FIGURES.items():
        path = STATIC / f"{stem}.svg"
        save_svg(draw(snapshot), path)
        print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
