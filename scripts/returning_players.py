"""Next-season check of the box-score prior on returning players.

Thin wrapper over ``athletevalue.valuation.checks.returning_players``, which the
extended validation also runs as a gate (``athletevalue validate --extended``).

    uv run python scripts/returning_players.py 2025 2026
"""

from __future__ import annotations

import sys

from athletevalue.api import Session
from athletevalue.valuation.checks import returning_players


def main(first: int, second: int) -> None:
    session = Session()
    registry = session.registry
    earlier = session.fit_season(first)
    result = returning_players(
        earlier,
        session.fit_season(second),
        min_possessions_earlier=int(
            registry.get("mbb.prior.returning_min_possessions_earlier").scalar()
        ),
        min_possessions_later=int(
            registry.get("mbb.prior.returning_min_possessions_later").scalar()
        ),
    )
    if result is None:
        print("person ids are missing for one of the seasons")
        return
    trained = earlier.prior.train_seasons if earlier.prior else None
    print(f"{first} prior trained on: {trained}")
    print(
        f"{result.n_players} returning players: correlation with {second} no-prior ratings "
        f"{result.corr_with_prior:.3f} with prior, {result.corr_without_prior:.3f} without"
    )


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
