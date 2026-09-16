from __future__ import annotations

import polars as pl

from athletevalue.frames.games import build_season_games


def test_records_bids_and_neutral_sites():
    schedule = pl.DataFrame(
        {
            "contest_id": ["c1", "c2", "c3"],
            "game_date": ["11/01/2025", "03/20/2026", "03/22/2026"],
            "home": ["A", "A", "B"],
            "away": ["B", "C", "A"],
            "home_score": [70, 80, 60],
            "away_score": [65, 70, 75],
        }
    )
    espn = pl.DataFrame(
        {
            "game_id": [11, 12, 13],
            "game_date": [None, None, None],
            "home_location": ["A", "A", "B"],
            "away_location": ["B", "C", "A"],
            "neutral_site": [False, True, True],
            "season_type": [2, 3, 3],
            "tournament_id": [None, 22, 22],
            "home_id": [1, 1, 2],
            "away_id": [2, 3, 1],
            "home_score": [70, 80, 60],
            "away_score": [65, 70, 75],
        }
    )
    team_ids = pl.DataFrame(
        {"team": ["A", "B", "C"], "conference": ["X", "X", "Y"], "season": [2026] * 3}
    )
    game_map = pl.DataFrame({"contest_id": ["c1", "c2", "c3"], "espn_game_id": ["11", "12", "13"]})
    result = build_season_games(schedule, espn, team_ids, game_map)
    teams = {row["team"]: row for row in result.teams.iter_rows(named=True)}
    assert (teams["A"]["wins"], teams["A"]["losses"]) == (3, 0)
    assert teams["A"]["ncaa_bid"] and teams["A"]["ncaa_games"] == 2 and teams["A"]["ncaa_wins"] == 2
    assert teams["B"]["ncaa_bid"] and teams["B"]["ncaa_wins"] == 0
    assert result.games.filter(pl.col("neutral")).height == 2
