"""A number with its interval, unit, method and evidence status."""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from athletevalue.schemas.evidence import EvidenceStatus

_TOLERANCE = 1e-9


class Estimate(BaseModel):
    """A point value and a central interval at probability ``level``.

    ``value`` is the median for simulated quantities and the mean for normal
    approximations; both lie inside the interval by construction.
    """

    model_config = ConfigDict(frozen=True)

    value: float
    lower: float
    upper: float
    level: float = Field(default=0.8, gt=0.0, lt=1.0)
    unit: str
    status: EvidenceStatus
    method: str
    se: float | None = None

    @model_validator(mode="after")
    def _check_interval(self) -> Self:
        for name in ("value", "lower", "upper"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{self.method}: {name} must be finite")
        scale = _TOLERANCE * max(1.0, abs(self.value))
        if not (self.lower - scale <= self.value <= self.upper + scale):
            raise ValueError(
                f"{self.method}: value {self.value} outside interval [{self.lower}, {self.upper}]"
            )
        return self

    @classmethod
    def exact(cls, value: float, *, unit: str, status: EvidenceStatus, method: str) -> Self:
        return cls(value=value, lower=value, upper=value, unit=unit, status=status, method=method)

    @classmethod
    def normal(
        cls,
        mean: float,
        se: float,
        *,
        unit: str,
        status: EvidenceStatus,
        method: str,
        level: float = 0.8,
    ) -> Self:
        z = NormalDist().inv_cdf(0.5 + level / 2)
        return cls(
            value=mean,
            lower=mean - z * se,
            upper=mean + z * se,
            level=level,
            unit=unit,
            status=status,
            method=method,
            se=se,
        )

    @property
    def width(self) -> float:
        return self.upper - self.lower
