"""Next-season check of the box-score prior on returning players.

stats.ncaa.org player ids are season-scoped, so players are linked across seasons
with the ``person_id`` column of the SportsDataverse reference RAPM release. The
earlier season is fitted with its default prior (earlier seasons only), and its
ratings are correlated with the later season's no-prior ratings.

    uv run python scripts/returning_players.py 2025 2026
"""

from __future__ import annotations

import sys

import numpy as np
import polars as pl

from athletevalue import api
from athletevalue.sources.cache import ArtifactCache
from athletevalue.sources.sportsdataverse import SdvClient, SdvDataset


def main(first: int, second: int, min_poss_first: int = 200, min_poss_second: int = 500) -> None:
    client = SdvClient(ArtifactCache.default())
    ids = {
        season: client.frame(SdvDataset.REFERENCE_RAPM, season).select(
            pl.col("player_id").cast(pl.Utf8).alias("athlete_id"), "person_id"
        )
        for season in (first, second)
    }
    earlier = api.fit_season(first)
    later = api.fit_season(second, prior=False)
    target = (
        later.baseline.table.filter(pl.col("off_poss") >= min_poss_second)
        .join(ids[second], on="athlete_id")
        .select("person_id", pl.col("net").alias("net_later"), pl.col("off_poss").alias("w"))
    )
    print(f"{first} prior trained on: {earlier.prior.train_seasons if earlier.prior else None}")
    for label, table in (("no prior", earlier.baseline.table), ("box prior", earlier.rapm.table)):
        joined = (
            table.filter(pl.col("off_poss") >= min_poss_first)
            .join(ids[first], on="athlete_id")
            .select("person_id", "net")
            .join(target, on="person_id")
        )
        x, y, w = joined["net"].to_numpy(), joined["net_later"].to_numpy(), joined["w"].to_numpy()
        slope, intercept = np.polyfit(x, y, 1, w=np.sqrt(w))
        r2 = 1 - np.sum(w * (y - (slope * x + intercept)) ** 2) / np.sum(
            w * (y - np.average(y, weights=w)) ** 2
        )
        print(
            f"{first} {label:<10} -> {second} no-prior net: {joined.height} returning players, "
            f"correlation {np.corrcoef(x, y)[0, 1]:.3f}, possession-weighted R^2 {r2:.3f}"
        )


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
