"""Finding a player by name."""

from __future__ import annotations

from typing import Any

import polars as pl

from athletevalue.identity.names import normalize_name, similarity

_MATCH_FLOOR = 0.6


class PlayerNotFoundError(LookupError):
    pass


class AmbiguousPlayerError(LookupError):
    pass


def resolve_player(table: pl.DataFrame, query: str, *, team: str | None = None) -> dict[str, Any]:
    """Return the row of *table* (athlete_id, name, team, ...) matching *query*.

    An exact normalized name match wins. Otherwise the closest names above a
    similarity floor are candidates; more than one best candidate is an error that
    lists them, so a caller can add ``team``.
    """
    candidates = table
    if team is not None:
        wanted = normalize_name(team)
        candidates = candidates.filter(
            pl.col("team").map_elements(normalize_name, return_dtype=pl.Utf8) == wanted
        )
        if candidates.is_empty():
            raise PlayerNotFoundError(f"no team named {team!r} in this season")
    target = normalize_name(query)
    exact = candidates.filter(
        pl.col("name").map_elements(normalize_name, return_dtype=pl.Utf8) == target
    )
    pool = exact if not exact.is_empty() else candidates
    if exact.is_empty():
        scored = pool.with_columns(
            pl.col("name")
            .map_elements(lambda n: similarity(n, query), return_dtype=pl.Float64)
            .alias("_score")
        ).filter(pl.col("_score") >= _MATCH_FLOOR)
        if scored.is_empty():
            raise PlayerNotFoundError(f"no player close to {query!r}")
        best = scored["_score"].max()
        pool = scored.filter(pl.col("_score") == best).drop("_score")
    if pool.height > 1:
        options = ", ".join(f"{r['name']} ({r['team']})" for r in pool.iter_rows(named=True))
        raise AmbiguousPlayerError(f"{query!r} matches several players: {options}. Pass a team.")
    return pool.row(0, named=True)
