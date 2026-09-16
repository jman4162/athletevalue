"""Lineup rows to a sparse ridge design matrix.

Columns, in order: one offensive column per modelled player, one defensive
column per modelled player, then pooled-offense, pooled-defense,
non-D1-offense, non-D1-defense, home court and intercept.

Model for a row with offense lineup O and defense lineup D:

    100 * pts / poss = intercept + h * venue + sum_{i in O} off_i - sum_{j in D} def_j

so a positive ``def_j`` means player j lowers opponent scoring, and a player's
net rating is ``off + def``. ``venue`` is +1 when the offense is at home, -1
when away and 0 at a neutral site; the home team's net advantage is ``2h``.
Players below the possession threshold share the pooled columns. A non-D1
opponent's lineup has no player ids and enters as a single column.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
import scipy.sparse as sp
from numpy.typing import NDArray

from athletevalue.constants import PER_100
from athletevalue.frames.possessions import LineupData


@dataclass(frozen=True)
class Design:
    X: sp.csr_matrix
    y: NDArray[np.float64]
    w: NDArray[np.float64]
    groups: NDArray[np.int64]
    athlete_ids: tuple[str, ...]
    pooled_ids: frozenset[str]
    penalized: NDArray[np.bool_]

    @property
    def n_players(self) -> int:
        return len(self.athlete_ids)

    @property
    def n_columns(self) -> int:
        return int(self.X.shape[1])

    def off_col(self, index: int) -> int:
        return index

    def def_col(self, index: int) -> int:
        return self.n_players + index

    def _extra(self, name: str) -> int:
        return 2 * self.n_players + EXTRA_COLUMNS.index(name)

    @property
    def pool_off_col(self) -> int:
        return self._extra("pool_off")

    @property
    def pool_def_col(self) -> int:
        return self._extra("pool_def")

    @property
    def non_d1_off_col(self) -> int:
        return self._extra("non_d1_off")

    @property
    def non_d1_def_col(self) -> int:
        return self._extra("non_d1_def")

    @property
    def home_col(self) -> int:
        return self._extra("home")

    @property
    def intercept_col(self) -> int:
        return self._extra("intercept")

    def offense_columns(self) -> NDArray[np.int64]:
        return np.append(np.arange(self.n_players), self.pool_off_col)

    def defense_columns(self) -> NDArray[np.int64]:
        return np.append(np.arange(self.n_players, 2 * self.n_players), self.pool_def_col)


EXTRA_COLUMNS = ("pool_off", "pool_def", "non_d1_off", "non_d1_def", "home", "intercept")
"""Unmodelled-player, venue and intercept columns, in order after the player blocks."""


def build_design(lineups: LineupData, *, min_possessions: float) -> Design:
    rows = lineups.rows.with_row_index("row")
    players = lineups.players.with_columns(
        ((pl.col("off_poss") + pl.col("def_poss")) / 2).alias("_avg_poss")
    )
    modelled = players.filter(pl.col("_avg_poss") >= min_possessions).sort("athlete_id")
    athlete_ids = tuple(modelled["athlete_id"].to_list())
    pooled = frozenset(players.filter(pl.col("_avg_poss") < min_possessions)["athlete_id"])
    n_players = len(athlete_ids)
    n_columns = 2 * n_players + len(EXTRA_COLUMNS)
    extra = {name: 2 * n_players + k for k, name in enumerate(EXTRA_COLUMNS)}
    pool_off, pool_def = extra["pool_off"], extra["pool_def"]
    non_d1_off, non_d1_def = extra["non_d1_off"], extra["non_d1_def"]
    home, intercept = extra["home"], extra["intercept"]

    index = pl.DataFrame(
        {"athlete_id": list(athlete_ids), "player_index": np.arange(n_players, dtype=np.int64)},
        schema={"athlete_id": pl.Utf8, "player_index": pl.Int64},
    )

    def player_entries(
        list_column: str, offset: int, pool_column: int, sign: float
    ) -> pl.DataFrame:
        exploded = (
            rows.select("row", pl.col(list_column).alias("athlete_id"))
            .explode("athlete_id", empty_as_null=False)
            .drop_nulls("athlete_id")
            .join(index, on="athlete_id", how="left")
        )
        return exploded.select(
            pl.col("row").cast(pl.Int64),
            pl.when(pl.col("player_index").is_null())
            .then(pool_column)
            .otherwise(pl.col("player_index") + offset)
            .cast(pl.Int64)
            .alias("col"),
            pl.lit(sign).alias("val"),
        )

    parts = [
        player_entries("off_ids", 0, pool_off, 1.0),
        player_entries("def_ids", n_players, pool_def, -1.0),
        rows.filter(~pl.col("offense_d1")).select(
            pl.col("row").cast(pl.Int64),
            pl.lit(non_d1_off, pl.Int64).alias("col"),
            pl.lit(1.0).alias("val"),
        ),
        rows.filter(~pl.col("defense_d1")).select(
            pl.col("row").cast(pl.Int64),
            pl.lit(non_d1_def, pl.Int64).alias("col"),
            pl.lit(-1.0).alias("val"),
        ),
        rows.filter(pl.col("venue") != 0).select(
            pl.col("row").cast(pl.Int64),
            pl.lit(home, pl.Int64).alias("col"),
            pl.col("venue").cast(pl.Float64).alias("val"),
        ),
        rows.select(
            pl.col("row").cast(pl.Int64),
            pl.lit(intercept, pl.Int64).alias("col"),
            pl.lit(1.0).alias("val"),
        ),
    ]
    entries = pl.concat(parts)
    X = sp.coo_matrix(
        (
            entries["val"].to_numpy(),
            (entries["row"].to_numpy(), entries["col"].to_numpy()),
        ),
        shape=(rows.height, n_columns),
    ).tocsr()
    X.sum_duplicates()

    poss = rows["poss"].cast(pl.Float64).to_numpy()
    y = PER_100 * rows["pts"].cast(pl.Float64).to_numpy() / poss
    groups = (
        rows.select(pl.col("contest_id").cast(pl.Categorical).to_physical())["contest_id"]
        .to_numpy()
        .astype(np.int64)
    )
    penalized = np.zeros(n_columns, dtype=bool)
    penalized[: 2 * n_players] = True
    return Design(
        X=X,
        y=y,
        w=poss,
        groups=groups,
        athlete_ids=athlete_ids,
        pooled_ids=pooled,
        penalized=penalized,
    )
