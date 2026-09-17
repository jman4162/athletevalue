"""Render the README example on a synthetic season.

The README does not name a real athlete. This builds the same synthetic season the
offline tests use, values its best player, relabels him and prints the summary.

    uv run python scripts/readme_example.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.fixtures.example_program import example_player, example_program

from athletevalue.schemas.identity import PlayerRef
from athletevalue.valuation.player import value_player


def main() -> None:
    season, economics, registry = example_program()
    best = example_player(season)
    valuation = value_player(
        season, economics, registry, best["name"], seed=0, as_of=date(2026, 9, 17)
    )
    relabelled = valuation.model_copy(
        update={
            "player": PlayerRef(athlete_id="example", name="A. Guard", team="State U", season=2026),
            "conference": "Example Conf",
        }
    )
    print(relabelled.summary().replace(best["team"], "State U").replace("Big Ten", "Example Conf"))


if __name__ == "__main__":
    main()
