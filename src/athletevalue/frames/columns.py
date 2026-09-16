"""Column contracts for the SportsDataverse inputs this package reads.

Checking these up front turns an upstream schema change into one clear error
instead of a confusing failure deep in a fit.
"""

from __future__ import annotations

import polars as pl

SLOTS = tuple(range(1, 6))

POSSESSION_COLUMNS: tuple[str, ...] = (
    "contest_id",
    "home",
    "away",
    "poss_team",
    "pts",
    "is_garbage_time",
    "espn_game_id",
    "home_espn_team_id",
    "away_espn_team_id",
    *(f"{side}_{slot}_player_id" for side in ("home", "away") for slot in SLOTS),
    *(f"{side}_{slot}_clean_name" for side in ("home", "away") for slot in SLOTS),
)
TEAM_ID_COLUMNS: tuple[str, ...] = ("team", "conference", "season")
SCHEDULE_COLUMNS: tuple[str, ...] = (
    "contest_id",
    "game_date",
    "home",
    "away",
    "home_score",
    "away_score",
)
ESPN_SCHEDULE_COLUMNS: tuple[str, ...] = (
    "game_id",
    "game_date",
    "home_location",
    "away_location",
    "neutral_site",
    "season_type",
    "tournament_id",
    "home_id",
    "away_id",
    "home_score",
    "away_score",
)
REFERENCE_RAPM_COLUMNS: tuple[str, ...] = ("player_id", "player", "team", "rapm_net", "off_poss")

NCAA_TOURNAMENT_ID = 22
"""ESPN's tournament id for the NCAA men's championship; constant in 2012-2026 files."""

POSTSEASON_TYPE = 3

MATCH_MIN_SIMILARITY = 0.6
"""Average name similarity both teams need when matching an ESPN game to an NCAA game."""

MATCH_DAY_WINDOW = 1
"""ESPN and stats.ncaa.org dates for the same game can differ by a day near midnight."""


class SchemaError(ValueError):
    pass


def require_columns(frame: pl.DataFrame, columns: tuple[str, ...], name: str) -> pl.DataFrame:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise SchemaError(f"{name} is missing columns {missing}")
    return frame
