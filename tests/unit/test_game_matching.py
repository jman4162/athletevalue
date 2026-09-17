from __future__ import annotations

from datetime import date

import polars as pl

from athletevalue.frames.games import combined_game_map, match_espn_games


def test_matches_by_date_window_and_names_including_swapped_venue():
    schedule = pl.DataFrame(
        {
            "contest_id": ["c1", "c2", "c3"],
            "game_date": ["03/20/2015", "03/21/2015", "11/10/2014"],
            "home": ["Michigan St.", "Fla. Atlantic", "Duke"],
            "away": ["Virginia", "Miami (FL)", "Army West Point"],
            "home_score": [1, 1, 1],
            "away_score": [0, 0, 0],
        }
    )
    espn = pl.DataFrame(
        {
            "game_id": [100, 200, 300],
            "game_date": [date(2015, 3, 21), date(2015, 3, 21), date(2014, 11, 10)],
            "home_location": ["Michigan State", "Miami", "Duke"],
            "away_location": ["Virginia", "Florida Atlantic", "Army"],
            "neutral_site": [True, True, False],
            "season_type": [3, 3, 2],
        }
    )
    matched = dict(match_espn_games(schedule, espn).iter_rows())
    assert matched == {"c1": "100", "c2": "200"}


def test_combined_map_prefers_possession_ids():
    possessions = pl.DataFrame({"contest_id": ["c1", "c1"], "espn_game_id": ["999", "999"]})
    schedule = pl.DataFrame(
        {
            "contest_id": ["c1"],
            "game_date": ["03/20/2015"],
            "home": ["A"],
            "away": ["B"],
            "home_score": [1],
            "away_score": [0],
        }
    )
    espn = pl.DataFrame(
        {
            "game_id": [999],
            "game_date": [date(2015, 3, 20)],
            "home_location": ["A"],
            "away_location": ["B"],
            "neutral_site": [True],
            "season_type": [3],
        }
    )
    assert combined_game_map(possessions, schedule, espn).to_dicts() == [
        {"contest_id": "c1", "espn_game_id": "999"}
    ]


def test_abbreviated_names_match_through_crosswalk_aliases():
    schedule = pl.DataFrame(
        {
            "contest_id": ["c1", "c2"],
            "game_date": ["03/21/2019", "03/22/2018"],
            "home": ["Gonzaga", "Southern California"],
            "away": ["FDU", "UNI"],
            "home_score": [1, 1],
            "away_score": [0, 0],
        }
    )
    espn = pl.DataFrame(
        {
            "game_id": [100, 200],
            "game_date": [date(2019, 3, 21), date(2018, 3, 22)],
            "home_location": ["Gonzaga", "USC"],
            "away_location": ["Fairleigh Dickinson", "Northern Iowa"],
            "neutral_site": [True, True],
            "season_type": [3, 2],
        }
    )
    assert dict(match_espn_games(schedule, espn).iter_rows()) == {"c1": "100", "c2": "200"}
    assert match_espn_games(schedule, espn, aliases={}).is_empty()
