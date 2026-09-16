"""Readers for files a user supplies from their own licensed accounts.

KenPom and the Knight-Newhouse College Athletics Database restrict
redistribution and commercial use. This package never downloads them and never
uses them in fitting code; an import-linter contract enforces the second rule.
These readers exist so a subscriber can compare outputs locally.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl


class UserFileError(ValueError):
    pass


KENPOM_REQUIRED = ("team", "adj_o", "adj_d")


def read_kenpom_team_ratings(path: Path) -> pl.DataFrame:
    """Read a user's own KenPom team-ratings export with columns team, adj_o, adj_d."""
    frame = pl.read_csv(path)
    missing = [column for column in KENPOM_REQUIRED if column not in frame.columns]
    if missing:
        raise UserFileError(f"{path}: missing columns {missing}; expected {KENPOM_REQUIRED}")
    return frame.select(
        pl.col("team").cast(pl.Utf8),
        pl.col("adj_o").cast(pl.Float64),
        pl.col("adj_d").cast(pl.Float64),
    ).with_columns((pl.col("adj_o") - pl.col("adj_d")).alias("adj_em"))
