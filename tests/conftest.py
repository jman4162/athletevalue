from __future__ import annotations

import pytest

from athletevalue.assumptions.registry import AssumptionRegistry


@pytest.fixture(scope="session")
def registry() -> AssumptionRegistry:
    return AssumptionRegistry.load()
