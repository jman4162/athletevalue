"""Identifiers for players and teams."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class PlayerRef(BaseModel):
    """A player in one season.

    ``athlete_id`` is the stats.ncaa.org player id carried by the SportsDataverse
    possession files. It is stable within a season; linking across seasons is
    not attempted in v0.1.
    """

    model_config = ConfigDict(frozen=True)

    athlete_id: str
    name: str
    team: str
    season: int


class TeamRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    team: str
    season: int
    conference: str | None = None
