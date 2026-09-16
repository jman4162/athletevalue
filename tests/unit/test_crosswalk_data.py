from __future__ import annotations

import polars as pl

from athletevalue.identity.crosswalk import team_crosswalk


def test_crosswalk_is_well_formed():
    crosswalk = team_crosswalk()
    assert crosswalk.columns == ["team", "unitid", "eada_institution_name"]
    assert crosswalk.height > 360
    assert crosswalk.filter(pl.col("unitid").is_null())["team"].to_list() == [
        "Air Force",
        "Army West Point",
        "Navy",
        "New Haven",
    ]
    assert crosswalk.unique(subset=["team", "unitid"]).height == crosswalk.height
    michigan = crosswalk.filter(pl.col("team") == "Michigan").row(0, named=True)
    assert michigan["eada_institution_name"] == "University of Michigan-Ann Arbor"
