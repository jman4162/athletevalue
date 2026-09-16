"""Download and read one season of men's basketball inputs."""

from __future__ import annotations

from athletevalue.sources.cache import ArtifactCache
from athletevalue.sources.sportsdataverse import SdvClient, SdvDataset
from athletevalue.valuation.season import SeasonFrames

_INPUTS = (
    SdvDataset.POSSESSIONS,
    SdvDataset.TEAM_IDS,
    SdvDataset.SCHEDULE,
    SdvDataset.ESPN_SCHEDULE,
)


def load_season_frames(season: int, cache: ArtifactCache) -> SeasonFrames:
    client = SdvClient(cache)
    return SeasonFrames(
        season=season,
        possessions=client.frame(SdvDataset.POSSESSIONS, season),
        team_ids=client.frame(SdvDataset.TEAM_IDS, season),
        schedule=client.frame(SdvDataset.SCHEDULE, season),
        espn_schedule=client.frame(SdvDataset.ESPN_SCHEDULE, season),
        sources=tuple(client.reference(dataset, season) for dataset in _INPUTS),
    )
