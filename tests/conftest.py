from __future__ import annotations

import pytest

from athletevalue.assumptions.registry import AssumptionRegistry
from tests.fixtures.cache_dir import populate_cache


@pytest.fixture(scope="session")
def registry() -> AssumptionRegistry:
    return AssumptionRegistry.load()


ONE_PRIOR_SEASON = """\
["mbb.prior.training_seasons"]
description = "Train the prior on the one synthetic season before the target."
unit = "count"
basis = "user_input"
status = "scenario"
value = 1.0
"""


@pytest.fixture(scope="session")
def synthetic_cache_dir(tmp_path_factory):
    return populate_cache(tmp_path_factory.mktemp("athletevalue-cache"))


@pytest.fixture(scope="session")
def one_prior_season(tmp_path_factory):
    path = tmp_path_factory.mktemp("assumptions") / "one_prior_season.toml"
    path.write_text(ONE_PRIOR_SEASON, encoding="utf-8")
    return path
