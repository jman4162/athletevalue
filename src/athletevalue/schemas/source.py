"""Citations and licenses for the evidence behind a number."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

from athletevalue.schemas.hashing import stable_id


class SourceKind(StrEnum):
    DATA_RELEASE = "data_release"
    GOVERNMENT_DATA = "government_data"
    ACADEMIC_PAPER = "academic_paper"
    PRESS = "press"
    METHODOLOGY = "methodology"
    USER = "user"


class LicenseTag(StrEnum):
    """Redistribution terms of a source, as far as this project relies on them."""

    PUBLIC_DOMAIN = "public_domain"
    MIT = "mit"
    NCAA_DERIVED = "ncaa_derived"
    """Parsed from stats.ncaa.org play-by-play by SportsDataverse and published from an
    MIT-licensed repository. The MIT grant covers that compilation, not rights the NCAA
    may hold in the underlying data; the origin's terms restrict automated access."""
    ESPN_DERIVED = "espn_derived"
    """Collected from ESPN's public API by SportsDataverse and published from an
    MIT-licensed repository, on the same footing as ``NCAA_DERIVED``."""
    CC_BY_4 = "cc_by_4"
    CC0 = "cc0"
    FACTUAL_CITATION = "factual_citation"
    """A fact quoted from a press or methodology page, with attribution."""
    VALIDATION_ONLY = "validation_only"
    """Readable for comparison; never bundled, never used to fit shipped values."""
    RESTRICTED_USER_SUPPLIED = "restricted_user_supplied"
    """Supplied by a user from their own licensed account. Never distributed."""

    @property
    def may_train_shipped_models(self) -> bool:
        return self in _TRAINABLE


_TRAINABLE = frozenset(
    {
        LicenseTag.PUBLIC_DOMAIN,
        LicenseTag.MIT,
        LicenseTag.NCAA_DERIVED,
        LicenseTag.ESPN_DERIVED,
        LicenseTag.CC_BY_4,
        LicenseTag.CC0,
        LicenseTag.FACTUAL_CITATION,
    }
)
"""Sources this package fits on. The two derived tags are included on the strength of
SportsDataverse's public release, not of any grant from the NCAA or ESPN; see
LICENSE-DATA for what that does and does not settle."""


class SourceReference(BaseModel):
    """A citation precise enough that a reader can check the number."""

    model_config = ConfigDict(frozen=True)

    source_id: str = ""
    kind: SourceKind
    title: str
    url: str | None = None
    section: str | None = None
    quote: str | None = None
    published: str | None = None
    archived_url: str | None = None
    """A Wayback Machine snapshot, for pages that block automated readers or may move."""
    primary_source: str | None = None
    """The original study or document when the cited page is a report of it."""
    retrieved_at: datetime | None = None
    content_sha256: str | None = None
    license: LicenseTag

    @model_validator(mode="before")
    @classmethod
    def _derive_id(cls, data: Any) -> Any:
        if isinstance(data, dict) and not data.get("source_id"):
            data = dict(data)
            data["source_id"] = stable_id(
                "src", data.get("kind"), data.get("url"), data.get("section"), data.get("quote")
            )
        return data

    @property
    def is_verifiable(self) -> bool:
        """Whether this points at a specific passage or artifact a reader can check."""
        return bool(self.quote or self.section or self.content_sha256)
