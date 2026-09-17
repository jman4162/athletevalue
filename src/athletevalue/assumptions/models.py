"""Assumptions carry their evidence, or they are not allowed into a model.

No model parameter may be a bare literal in a modelling package. Every default is
a registry entry with a declared basis, and the basis sets the evidence status of
everything computed from it.

Two bases need a word of explanation:

``modelling_choice`` covers tuning parameters of a statistical estimator (a
pooling threshold, a number of folds). They change how an answer is computed,
not what is assumed about the world, so they require a written rationale and
yield ``estimated`` status.

``user_input`` covers assumptions about the world that no public source pins
down, such as how a conference splits tournament money. Any stored value is a
starting point, and every result that depends on one is a ``scenario``.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from athletevalue.schemas.evidence import EvidenceStatus
from athletevalue.schemas.source import SourceReference

AssumptionValue = float | bool | str | tuple[float, ...] | tuple[str, ...]


class AssumptionBasis(StrEnum):
    PUBLISHED_RULE = "published_rule"
    """A rule or term stated by the body that sets it. Requires a citation."""

    PUBLISHED_THIRD_PARTY = "published_third_party"
    """A figure from a citable public source. Requires a citation with a URL."""

    DERIVED_FROM_DATA = "derived_from_data"
    """Estimated at runtime from the input data; carries no fixed value."""

    MODELLING_CHOICE = "modelling_choice"
    """A tuning parameter of an estimator. Requires a rationale."""

    USER_INPUT = "user_input"
    """A what-if supplied by whoever runs the model."""


_STATUS_BY_BASIS: dict[AssumptionBasis, EvidenceStatus] = {
    AssumptionBasis.PUBLISHED_RULE: EvidenceStatus.REPORTED,
    AssumptionBasis.PUBLISHED_THIRD_PARTY: EvidenceStatus.ESTIMATED,
    AssumptionBasis.DERIVED_FROM_DATA: EvidenceStatus.DERIVED,
    AssumptionBasis.MODELLING_CHOICE: EvidenceStatus.ESTIMATED,
    AssumptionBasis.USER_INPUT: EvidenceStatus.SCENARIO,
}


class Assumption(BaseModel):
    """One model parameter and the evidence for its value."""

    model_config = ConfigDict(frozen=True)

    assumption_id: str
    description: str
    unit: str
    basis: AssumptionBasis
    value: AssumptionValue | None = None
    citation: SourceReference | None = None
    rationale: str | None = None

    @property
    def status(self) -> EvidenceStatus:
        return _STATUS_BY_BASIS[self.basis]

    @model_validator(mode="after")
    def _check_evidence(self) -> Self:
        aid = self.assumption_id
        if self.basis in (AssumptionBasis.PUBLISHED_RULE, AssumptionBasis.PUBLISHED_THIRD_PARTY):
            if self.citation is None or not self.citation.url:
                raise ValueError(f"{aid}: {self.basis} requires a citation with a URL")
            if not self.citation.is_verifiable:
                raise ValueError(f"{aid}: citation needs a quote or section a reader can check")
            if self.value is None:
                raise ValueError(f"{aid}: {self.basis} requires a value")
        elif self.basis is AssumptionBasis.DERIVED_FROM_DATA:
            if self.value is not None:
                raise ValueError(f"{aid}: derived_from_data must not pin a value")
        elif self.basis is AssumptionBasis.MODELLING_CHOICE:
            if not self.rationale:
                raise ValueError(f"{aid}: modelling_choice requires a rationale")
            if self.value is None:
                raise ValueError(f"{aid}: modelling_choice requires a value")
        elif self.value is None:
            raise ValueError(f"{aid}: user_input requires a starting value")
        return self

    def scalar(self) -> float:
        if isinstance(self.value, bool) or not isinstance(self.value, int | float):
            raise TypeError(f"{self.assumption_id} is not a scalar: {self.value!r}")
        return float(self.value)

    def flag(self) -> bool:
        if not isinstance(self.value, bool):
            raise TypeError(f"{self.assumption_id} is not a flag: {self.value!r}")
        return self.value

    def interval(self) -> tuple[float, float]:
        value = self.value
        if not isinstance(value, tuple) or len(value) != 2:
            raise TypeError(f"{self.assumption_id} is not a (low, high) pair: {value!r}")
        low, high = value
        if not isinstance(low, int | float) or not isinstance(high, int | float) or low > high:
            raise TypeError(f"{self.assumption_id} is not an ordered numeric pair: {value!r}")
        return float(low), float(high)

    def numbers(self) -> tuple[float, ...]:
        value = self.value
        if not isinstance(value, tuple) or not all(isinstance(v, int | float) for v in value):
            raise TypeError(f"{self.assumption_id} is not a numeric list: {value!r}")
        return tuple(float(v) for v in value)

    def text(self) -> str:
        if not isinstance(self.value, str):
            raise TypeError(f"{self.assumption_id} is not a text choice: {self.value!r}")
        return self.value

    def names(self) -> tuple[str, ...]:
        value = self.value
        if not isinstance(value, tuple) or not all(isinstance(v, str) for v in value):
            raise TypeError(f"{self.assumption_id} is not a list of names: {value!r}")
        return tuple(str(v) for v in value)
