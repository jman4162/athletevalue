from __future__ import annotations

import pytest
from pydantic import ValidationError

from athletevalue.schemas.estimate import Estimate
from athletevalue.schemas.evidence import EvidenceStatus
from athletevalue.schemas.source import LicenseTag


def test_weakest_status_wins():
    assert (
        EvidenceStatus.weakest(
            EvidenceStatus.REPORTED, EvidenceStatus.SCENARIO, EvidenceStatus.DERIVED
        )
        is EvidenceStatus.SCENARIO
    )
    assert EvidenceStatus.weakest() is EvidenceStatus.REPORTED


def test_estimate_rejects_value_outside_interval():
    with pytest.raises(ValidationError):
        Estimate(
            value=5, lower=0, upper=4, unit="wins", status=EvidenceStatus.ESTIMATED, method="x"
        )


def test_normal_estimate_interval_matches_level():
    est = Estimate.normal(
        10, 2, unit="wins", status=EvidenceStatus.ESTIMATED, method="x", level=0.8
    )
    assert est.lower == pytest.approx(10 - 1.2816 * 2, abs=1e-3)
    assert est.upper == pytest.approx(10 + 1.2816 * 2, abs=1e-3)


def test_only_redistributable_licenses_may_train():
    assert LicenseTag.MIT.may_train_shipped_models
    assert LicenseTag.PUBLIC_DOMAIN.may_train_shipped_models
    assert not LicenseTag.VALIDATION_ONLY.may_train_shipped_models
    assert not LicenseTag.RESTRICTED_USER_SUPPLIED.may_train_shipped_models
