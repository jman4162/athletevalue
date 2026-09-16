"""One disclosed compensation transaction in the community deal registry."""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class DealType(StrEnum):
    REVENUE_SHARE = "revenue_share"
    COLLECTIVE = "collective"
    ENDORSEMENT = "endorsement"
    APPEARANCE = "appearance"
    OTHER = "other"


class SourceQuality(StrEnum):
    """How directly the dollar figure was observed. Strongest first."""

    CONTRACT_OR_RECORDS_REQUEST = "contract_or_records_request"
    NAMED_REPORT = "named_report"
    """A named journalist reporting a specific figure for a specific athlete."""
    ANONYMOUS_REPORT = "anonymous_report"


class DealRecord(BaseModel):
    """A deal with a public source.

    Valuations published by rating sites (On3, The NIL Standard, Opendorse) are
    model outputs, not transactions, and are rejected by the contribution rules
    rather than by this schema; ``source_quality`` has no member for them.
    """

    model_config = ConfigDict(frozen=True)

    deal_id: str
    athlete_name: str
    athlete_id: str | None = None
    season: int = Field(ge=2022)
    school: str
    sport: str
    deal_date: date | None = None
    cash_value: float = Field(ge=0)
    in_kind_value: float = Field(default=0.0, ge=0)
    duration_months: float = Field(gt=0)
    counterparty: str | None = None
    deal_type: DealType
    deliverables: str | None = None
    source_url: HttpUrl
    source_quality: SourceQuality
    notes: str | None = None

    @property
    def annualized_value(self) -> float:
        months_per_year = 12
        return (self.cash_value + self.in_kind_value) * months_per_year / self.duration_months

    @model_validator(mode="after")
    def _check_sport(self) -> Self:
        if self.sport not in {"mbb", "wbb", "cfb"}:
            raise ValueError(f"{self.deal_id}: sport must be one of mbb, wbb, cfb")
        return self
