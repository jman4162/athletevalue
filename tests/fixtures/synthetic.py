"""Synthetic seasons with known player ratings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from athletevalue.frames.possessions import LineupData

POSSESSION_SD = 115.0
"""Per-possession SD of points x 100; sqrt of the ~13,300 residual variance seen in real data."""


@dataclass(frozen=True)
class SyntheticSeason:
    lineups: LineupData
    off: dict[str, float]
    defense: dict[str, float]
    intercept: float
    home: float


def make_season(
    *,
    n_teams: int = 12,
    roster: int = 8,
    games_per_pair: int = 4,
    stints_per_game: int = 24,
    poss_per_stint: int = 6,
    rating_sd: float = 3.0,
    intercept: float = 105.0,
    home: float = 3.0,
    noise_scale: float = 1.0,
    seed: int = 0,
    non_d1_games: int = 0,
    neutral_share: float = 0.0,
) -> SyntheticSeason:
    rng = np.random.default_rng(seed)
    teams = [f"T{t:02d}" for t in range(n_teams)]
    players = {team: [f"{team}p{k}" for k in range(roster)] for team in teams}
    off = {p: float(rng.normal(0, rating_sd)) for team in teams for p in players[team]}
    defense = {p: float(rng.normal(0, rating_sd)) for team in teams for p in players[team]}
    non_d1_net = -40.0

    rows: list[dict[str, object]] = []
    counts: dict[tuple[str, str], list[int]] = {}
    pairs = [(a, b) for a in teams for b in teams if a < b]
    schedule = [pair for pair in pairs for _ in range(games_per_pair)]
    schedule += [(teams[k % n_teams], "NOND1") for k in range(non_d1_games)]
    for game, (home_team, away_team) in enumerate(schedule, start=1):
        neutral = rng.random() < neutral_share
        for _ in range(stints_per_game):
            lineups = {}
            for team in (home_team, away_team):
                if team == "NOND1":
                    lineups[team] = []
                else:
                    lineups[team] = sorted(
                        rng.choice(players[team], size=5, replace=False).tolist()
                    )
            for offense, defense_team in ((home_team, away_team), (away_team, home_team)):
                venue = 0 if neutral else (1 if offense == home_team else -1)
                o_ids, d_ids = lineups[offense], lineups[defense_team]
                mean = intercept + home * venue
                mean += sum(off[p] for p in o_ids) - sum(defense[p] for p in d_ids)
                if offense == "NOND1":
                    mean += non_d1_net / 2
                if defense_team == "NOND1":
                    mean -= non_d1_net / 2
                poss = poss_per_stint
                rate = mean + rng.normal(0, noise_scale * POSSESSION_SD / np.sqrt(poss))
                rows.append(
                    {
                        "contest_id": f"g{game}",
                        "offense_team": offense,
                        "defense_team": defense_team,
                        "offense_d1": offense != "NOND1",
                        "defense_d1": defense_team != "NOND1",
                        "venue": venue,
                        "off_ids": o_ids,
                        "def_ids": d_ids,
                        "poss": poss,
                        "pts": rate * poss / 100,
                    }
                )
                for p in o_ids:
                    counts.setdefault((p, offense), [0, 0])[0] += poss
                for p in d_ids:
                    counts.setdefault((p, defense_team), [0, 0])[1] += poss
    frame = pl.DataFrame(
        rows,
        schema={
            "contest_id": pl.Utf8,
            "offense_team": pl.Utf8,
            "defense_team": pl.Utf8,
            "offense_d1": pl.Boolean,
            "defense_d1": pl.Boolean,
            "venue": pl.Int8,
            "off_ids": pl.List(pl.Utf8),
            "def_ids": pl.List(pl.Utf8),
            "poss": pl.UInt32,
            "pts": pl.Float64,
        },
    )
    player_table = pl.DataFrame(
        [
            {"athlete_id": p, "name": p, "team": team, "off_poss": o, "def_poss": d}
            for (p, team), (o, d) in counts.items()
        ]
    ).sort("athlete_id")
    lineups = LineupData(rows=frame, players=player_table, counts={"lineup_rows": frame.height})
    return SyntheticSeason(
        lineups=lineups, off=off, defense=defense, intercept=intercept, home=home
    )
