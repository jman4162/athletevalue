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
        LicenseTag.CC_BY_4,
        LicenseTag.CC0,
        LicenseTag.FACTUAL_CITATION,
    }
)


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
