"""A small season in the raw SportsDataverse column layout, with known player ratings."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from athletevalue.valuation.season import SeasonFrames

CONFERENCES = {"Big Ten": 4, "WCC": 4}
ROSTER = 9
POSSESSIONS_PER_GAME = 140
SEGMENT = 10
THREE_RATE = 0.12


def make_raw_season(seed: int = 0, season: int = 2026) -> tuple[SeasonFrames, dict[str, float]]:
    rng = np.random.default_rng(seed)
    teams: list[tuple[str, str]] = []
    for conference, count in CONFERENCES.items():
        teams += [(f"{conference[:3]} U{k}", conference) for k in range(count)]
    players = {
        team: [(f"{team}-{j}", f"{team.replace(' ', '')} Player{j}") for j in range(ROSTER)]
        for team, _ in teams
    }
    strength = {team: (2.0 if conf == "Big Ten" else -1.0) for team, conf in teams}
    net = {
        pid: float(rng.normal(strength[team], 3.0)) for team, _ in teams for pid, _ in players[team]
    }
    off = {pid: value / 2 for pid, value in net.items()}

    poss_rows: list[dict[str, object]] = []
    schedule_rows: list[dict[str, object]] = []
    espn_rows: list[dict[str, object]] = []
    day = date(2025, 11, 3)
    matchups = [(h, a) for h, _ in teams for a, _ in teams if h != a]
    for game_id, (home, away) in enumerate(matchups, start=1):
        day += timedelta(days=1)
        tournament = game_id > len(matchups) - 2
        score = {home: 0, away: 0}
        for poss in range(POSSESSIONS_PER_GAME):
            if poss % SEGMENT == 0:
                lineup = {
                    t: sorted(rng.choice(len(players[t]), 5, replace=False).tolist())
                    for t in (home, away)
                }
            offense = home if poss % 2 == 0 else away
            defense = away if offense == home else home
            venue = 0 if tournament else (1 if offense == home else -1)
            o_ids = [players[offense][k][0] for k in lineup[offense]]
            d_ids = [players[defense][k][0] for k in lineup[defense]]
            mean = (
                105 + 3 * venue + sum(off[p] for p in o_ids) - sum(net[p] - off[p] for p in d_ids)
            ) / 100
            two_rate = float(np.clip((mean - 3 * THREE_RATE) / 2, 0.05, 0.8))
            pts = int(rng.choice([0, 2, 3], p=[1 - two_rate - THREE_RATE, two_rate, THREE_RATE]))
            score[offense] += pts
            row: dict[str, object] = {
                "contest_id": f"c{game_id}",
                "home": home,
                "away": away,
                "poss_team": offense,
                "pts": pts,
                "is_garbage_time": 0,
                "espn_game_id": str(1000 + game_id),
                "home_espn_team_id": "1",
                "away_espn_team_id": "2",
            }
            for side, team in (("home", home), ("away", away)):
                for slot, k in enumerate(lineup[team], start=1):
                    row[f"{side}_{slot}_player_id"] = players[team][k][0]
                    row[f"{side}_{slot}_clean_name"] = players[team][k][1]
            poss_rows.append(row)
        schedule_rows.append(
            {
                "contest_id": f"c{game_id}",
                "game_date": day.strftime("%m/%d/%Y"),
                "home": home,
                "away": away,
                "home_score": score[home],
                "away_score": score[away],
                "season": season,
            }
        )
        espn_rows.append(
            {
                "game_id": 1000 + game_id,
                "game_date": day,
                "home_location": home,
                "away_location": away,
                "neutral_site": tournament,
                "season_type": 3 if tournament else 2,
                "tournament_id": 22 if tournament else None,
                "home_id": 1,
                "away_id": 2,
                "home_score": score[home],
                "away_score": score[away],
            }
        )
    box_rows = []
    for (athlete, team, contest), (poss, pts) in _box_counts(poss_rows).items():
        rating = off[athlete]
        box_rows.append(
            {
                "contest_id": contest,
                "player_id": athlete,
                "team": team,
                "o_poss": poss,
                "pts": pts,
                "fga": pts * 0.8,
                "tpa": poss * 0.1,
                "fta": poss * 0.05,
                "rima": poss * 0.08,
                "mida": poss * 0.05,
                "ast": max(0.0, poss * (0.05 + 0.01 * rating)),
                "tov": poss * 0.04,
                "orb": poss * 0.03,
                "drb": poss * 0.1,
                "stl": max(0.0, poss * (0.02 + 0.004 * (net[athlete] - rating))),
                "blk": poss * 0.01,
                "pf": poss * 0.05,
            }
        )
    frames = SeasonFrames(
        season=season,
        player_box=pl.DataFrame(box_rows),
        possessions=pl.DataFrame(poss_rows, infer_schema_length=None),
        team_ids=pl.DataFrame(
            {"team": [t for t, _ in teams], "conference": [c for _, c in teams], "season": season}
        ),
        schedule=pl.DataFrame(schedule_rows),
        espn_schedule=pl.DataFrame(espn_rows, schema_overrides={"tournament_id": pl.Int32}),
    )
    return frames, net


def _box_counts(
    poss_rows: list[dict[str, object]],
) -> dict[tuple[str, str, str], tuple[float, float]]:
    """Offensive possessions per player and game, with points split among the five."""
    counts: dict[tuple[str, str, str], list[float]] = {}
    for row in poss_rows:
        side = "home" if row["poss_team"] == row["home"] else "away"
        team = str(row[side])
        for slot in range(1, 6):
            key = (str(row[f"{side}_{slot}_player_id"]), team, str(row["contest_id"]))
            entry = counts.setdefault(key, [0.0, 0.0])
            entry[0] += 1
            entry[1] += float(row["pts"]) / 5  # type: ignore[arg-type]
    return {key: (value[0], value[1]) for key, value in counts.items()}
