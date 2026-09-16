from __future__ import annotations

import numpy as np
import polars as pl

from athletevalue.frames.possessions import LineupData
from athletevalue.impact.design import EXTRA_COLUMNS, build_design


def _lineups() -> LineupData:
    rows = pl.DataFrame(
        {
            "contest_id": ["g1", "g1", "g2"],
            "offense_team": ["A", "B", "A"],
            "defense_team": ["B", "A", "X"],
            "offense_d1": [True, True, True],
            "defense_d1": [True, True, False],
            "venue": [1, -1, 0],
            "off_ids": [
                ["a1", "a2", "a3", "a4", "a5"],
                ["b1", "b2", "b3", "b4", "b5"],
                ["a1", "a2", "a3", "a4", "a5"],
            ],
            "def_ids": [["b1", "b2", "b3", "b4", "b5"], ["a1", "a2", "a3", "a4", "a5"], []],
            "poss": [10, 8, 20],
            "pts": [12, 9, 30],
        },
        schema_overrides={"venue": pl.Int8, "poss": pl.UInt32, "def_ids": pl.List(pl.Utf8)},
    )
    ids = [f"a{k}" for k in range(1, 6)] + [f"b{k}" for k in range(1, 6)]
    players = pl.DataFrame(
        {
            "athlete_id": ids,
            "name": ids,
            "team": ["A"] * 5 + ["B"] * 5,
            "off_poss": [30] * 4 + [5] + [8] * 5,
            "def_poss": [8] * 4 + [5] + [10] * 5,
        }
    )
    return LineupData(rows=rows, players=players, counts={})


def test_columns_signs_and_pooling():
    design = build_design(_lineups(), min_possessions=9)
    # a5 averages 5 possessions and is pooled; everyone else has a column.
    assert design.pooled_ids == frozenset({"a5"})
    assert design.n_players == 9
    assert design.n_columns == 2 * 9 + len(EXTRA_COLUMNS)
    X = design.X.toarray()
    a1 = design.athlete_ids.index("a1")
    b1 = design.athlete_ids.index("b1")
    # Row 0: A on offense at home against B.
    assert X[0, design.off_col(a1)] == 1
    assert X[0, design.def_col(b1)] == -1
    assert X[0, design.pool_off_col] == 1
    assert X[0, design.home_col] == 1
    assert X[0, design.intercept_col] == 1
    # Row 1: B on offense away; pooled a5 is on defense.
    assert X[1, design.home_col] == -1
    assert X[1, design.pool_def_col] == -1
    # Row 2: neutral site against a non-D1 lineup.
    assert X[2, design.home_col] == 0
    assert X[2, design.non_d1_def_col] == -1
    np.testing.assert_allclose(design.y, [120.0, 112.5, 150.0])
    np.testing.assert_allclose(design.w, [10, 8, 20])
    assert design.groups[0] == design.groups[1] != design.groups[2]
    assert design.penalized.sum() == 18
    assert not design.penalized[design.intercept_col]
