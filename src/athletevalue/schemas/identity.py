"""Identifiers for players and teams."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class PlayerRef(BaseModel):
    """A player in one season.

    ``athlete_id`` is the stats.ncaa.org player id carried by the SportsDataverse
    possession files; it changes every season. ``person_id`` comes from the
    SportsDataverse reference RAPM release and follows the person across seasons;
    it is ``None`` when that release has no row for the player.
    """

    model_config = ConfigDict(frozen=True)

    athlete_id: str
    name: str
    team: str
    season: int
    person_id: str | None = None


class TeamRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    team: str
    season: int
    conference: str | None = None
