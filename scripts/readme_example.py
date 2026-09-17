"""Render the README example on a synthetic season.

The README does not name a real athlete. This builds the same synthetic season the
offline tests use, values its best player, relabels him and prints the summary.

    uv run python scripts/readme_example.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.fixtures.raw_season import make_raw_season

from athletevalue.assumptions.registry import default_registry
from athletevalue.economics.revenue import RevenueModel
from athletevalue.economics.tournament import BidModel
from athletevalue.frames.box import player_box_totals
from athletevalue.impact.prior import fit_box_prior
from athletevalue.schemas.identity import PlayerRef
from athletevalue.valuation.economy import EconomicsModel
from athletevalue.valuation.player import value_player
from athletevalue.valuation.season import assemble_season, baseline_rapm


def main() -> None:
    registry = default_registry()
    training, _ = make_raw_season(seed=1, season=2025)
    prior = fit_box_prior(
        [(2025, baseline_rapm(training, registry).table, player_box_totals(training.player_box))],
        pseudo_possessions=registry.get("mbb.prior.rate_pseudo_possessions").scalar(),
        ridge=registry.get("mbb.prior.box_ridge").scalar(),
    )
    frames, _ = make_raw_season(seed=0, season=2026)
    season = assemble_season(frames, registry, prior=prior)
    teams = season.teams["team"].to_list()
    economics = EconomicsModel(
        panel=pl.DataFrame(
            {"team": teams, "season": 2025, "rev_men": [12e6 if "Big" in t else 2e6 for t in teams]}
        ),
        outcomes=pl.DataFrame(),
        revenue=RevenueModel(
            point=np.zeros(4),
            draws=np.tile([0.004, 0.002, 0.03, 0.02], (20, 1)),
            n_obs=1,
            n_schools=1,
            first_season=2012,
            last_season=2025,
        ),
        bids=BidModel(coef=np.array([-10.0, 15.0, -6.0, 14.0]), n_obs=1),
        units_per_bid=1.9,
        power_conferences=frozenset({"Big Ten"}),
        reference_seasons=3,
        sources=(),
    )
    best = season.rapm.table.filter(~pl.col("pooled")).row(0, named=True)
    valuation = value_player(
        season, economics, registry, best["name"], seed=0, as_of=date(2026, 9, 17)
    )
    relabelled = valuation.model_copy(
        update={
            "player": PlayerRef(athlete_id="example", name="A. Guard", team="State U", season=2026),
            "conference": "Example Conf",
        }
    )
    print(relabelled.summary().replace(best["team"], "State U").replace("Big Ten", "Example Conf"))


if __name__ == "__main__":
    main()
