from __future__ import annotations

import polars as pl

from athletevalue.frames.possessions import build_lineup_data
from athletevalue.frames.team_possessions import team_possession_totals


def _possession(
    home, away, poss_team, pts, *, garbage=0, espn="e1", home_ids=None, away_ids=None, contest="c1"
):
    row = {
        "contest_id": contest,
        "home": home,
        "away": away,
        "poss_team": poss_team,
        "pts": pts,
        "is_garbage_time": garbage,
        "espn_game_id": espn,
        "home_espn_team_id": "1",
        "away_espn_team_id": "2",
    }
    for side, ids in (("home", home_ids), ("away", away_ids)):
        for slot in range(1, 6):
            value = None if ids is None else ids[slot - 1]
            row[f"{side}_{slot}_player_id"] = value
            row[f"{side}_{slot}_clean_name"] = None if value is None else f"Name {value}"
    return row


def _frame(rows):
    return pl.DataFrame(rows, infer_schema_length=None)


HOME = ["h5", "h4", "h3", "h2", "h1"]
AWAY = ["a1", "a2", "a3", "a4", "a5"]


def test_aggregates_by_lineup_offense_and_venue():
    rows = [
        _possession("A", "B", "A", 2, home_ids=HOME, away_ids=AWAY),
        _possession("A", "B", "A", 3, home_ids=list(reversed(HOME)), away_ids=AWAY),
        _possession("A", "B", "B", 0, home_ids=HOME, away_ids=AWAY),
        _possession("A", "B", "B", 2, garbage=1, home_ids=HOME, away_ids=AWAY),
    ]
    data = build_lineup_data(
        _frame(rows),
        d1_teams=frozenset({"A", "B"}),
        neutral_espn_games=frozenset(),
        drop_garbage_time=True,
    )
    assert data.counts["after_garbage_time"] == 3
    out = data.rows.sort("offense_team")
    assert out.height == 2
    first = out.row(0, named=True)
    assert first["offense_team"] == "A" and first["poss"] == 2 and first["pts"] == 5
    assert first["venue"] == 1
    assert first["off_ids"] == sorted(HOME)
    assert out.row(1, named=True)["venue"] == -1
    h1 = data.players.filter(pl.col("athlete_id") == "h1").row(0, named=True)
    assert (h1["off_poss"], h1["def_poss"], h1["team"]) == (2, 1, "A")


def test_non_d1_opponent_collapses_and_missing_d1_ids_are_dropped():
    rows = [
        _possession("A", "Tiny", "A", 2, home_ids=HOME, away_ids=None),
        _possession("A", "Tiny", "Tiny", 1, home_ids=HOME, away_ids=None),
        _possession("A", "B", "A", 2, home_ids=[*HOME[:4], None], away_ids=AWAY),
    ]
    data = build_lineup_data(
        _frame(rows),
        d1_teams=frozenset({"A", "B"}),
        neutral_espn_games=frozenset({"e1"}),
        drop_garbage_time=False,
    )
    assert data.counts["usable"] == 2
    assert set(data.rows["venue"]) == {0}
    tiny_offense = data.rows.filter(pl.col("offense_team") == "Tiny").row(0, named=True)
    assert tiny_offense["offense_d1"] is False and tiny_offense["off_ids"] == []
    assert "Tiny" not in set(data.players["team"])


def test_team_totals_include_garbage_time():
    rows = [
        _possession("A", "B", "A", 2, home_ids=HOME, away_ids=AWAY),
        _possession("A", "B", "B", 3, garbage=1, home_ids=HOME, away_ids=AWAY),
    ]
    totals = team_possession_totals(_frame(rows)).sort("team")
    a = totals.row(0, named=True)
    assert (a["off_poss"], a["pts_for"], a["def_poss"], a["pts_against"]) == (1, 2, 1, 3)
    assert a["ortg"] == 200.0 and a["pace"] == 1.0
