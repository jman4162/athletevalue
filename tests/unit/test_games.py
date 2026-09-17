from __future__ import annotations

import polars as pl

from athletevalue.constants import EVEN_WIN_PCT
from athletevalue.frames.games import build_season_games


def _toy_season() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
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
    return schedule, espn, team_ids, game_map


def test_records_bids_and_neutral_sites():
    result = build_season_games(*_toy_season())
    teams = {row["team"]: row for row in result.teams.iter_rows(named=True)}
    assert (teams["A"]["wins"], teams["A"]["losses"]) == (3, 0)
    assert teams["A"]["ncaa_bid"] and teams["A"]["ncaa_games"] == 2 and teams["A"]["ncaa_wins"] == 2
    assert teams["B"]["ncaa_bid"] and teams["B"]["ncaa_wins"] == 0
    assert result.games.filter(pl.col("neutral")).height == 2


def test_the_last_tournament_game_earns_no_unit():
    """The championship game is the one tournament game a conference is not paid for."""
    result = build_season_games(*_toy_season())
    teams = {row["team"]: row for row in result.teams.iter_rows(named=True)}
    # c3 on 03/22 is the latest tournament game, so A and B each played in the final.
    assert result.games.filter(pl.col("ncaa_final"))["contest_id"].to_list() == ["c3"]
    assert (teams["A"]["ncaa_games"], teams["A"]["ncaa_units"]) == (2, 1)
    assert (teams["B"]["ncaa_games"], teams["B"]["ncaa_units"]) == (1, 0)
    assert (teams["C"]["ncaa_games"], teams["C"]["ncaa_units"]) == (1, 1)


def test_schedule_strength_excludes_the_game_being_scored():
    """A beat B twice and C once; B and C lost every game they played.

    B's record without a given game against A is 0-1 or 0-0, so A's schedule rates
    at .000. A's record without either game is 2-0, so B and C face a 1.000 schedule.
    C played once, so it contributes no opponent record to A.
    """
    result = build_season_games(*_toy_season())
    sos = {row["team"]: row["sos"] for row in result.teams.iter_rows(named=True)}
    assert sos["A"] == 0.0
    assert sos["B"] == 1.0
    assert sos["C"] == 1.0


def test_a_team_with_no_measurable_opponent_gets_an_even_schedule():
    schedule = pl.DataFrame(
        {
            "contest_id": ["c1"],
            "game_date": ["11/01/2025"],
            "home": ["A"],
            "away": ["B"],
            "home_score": [70],
            "away_score": [65],
        }
    )
    espn = pl.DataFrame(
        {
            "game_id": [11],
            "game_date": [None],
            "home_location": ["A"],
            "away_location": ["B"],
            "neutral_site": [False],
            "season_type": [2],
            "tournament_id": [None],
            "home_id": [1],
            "away_id": [2],
            "home_score": [70],
            "away_score": [65],
        }
    )
    team_ids = pl.DataFrame({"team": ["A", "B"], "conference": ["X", "X"], "season": [2026] * 2})
    game_map = pl.DataFrame({"contest_id": ["c1"], "espn_game_id": ["11"]})
    teams = build_season_games(schedule, espn, team_ids, game_map).teams
    # Each played only the other, so neither opponent has another game to be judged on.
    assert teams["sos"].to_list() == [EVEN_WIN_PCT, EVEN_WIN_PCT]
    assert teams["ncaa_units"].to_list() == [0, 0]


def test_first_seen_codes_are_contiguous_in_order_of_appearance():
    from athletevalue.frames.codes import first_seen_codes

    # Categorical physical codes are shared across the process in recent polars, so
    # a cast elsewhere must not shift these codes.
    earlier = pl.Series(["zz", "yy", "c"]).cast(pl.Categorical)  # noqa: F841 (kept alive)
    values = pl.Series(["b", "a", "b", "c", "a"])
    assert first_seen_codes(values).tolist() == [0, 1, 0, 2, 1]
