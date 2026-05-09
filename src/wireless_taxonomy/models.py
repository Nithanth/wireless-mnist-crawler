from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


@dataclass(frozen=True)
class PaperSeed:
    title: str
    authors: list[str]
    venue: str
    year: int
    source_url: str
    abstract: str | None = None
    doi: str | None = None
    paper_url: str | None = None
    pdf_url: str | None = None
    session: str | None = None
    source_method: str = "unknown"
    source_confidence: float = 0.0
    evidence_text: str | None = None


@dataclass(frozen=True)
class PaperRecord(PaperSeed):
    id: int | None = None
    bibtex_key: str | None = None


@dataclass(frozen=True)
class EvidenceClaim:
    claim_id: str
    run_id: int | None
    claim_type: str
    claim_value: str
    evidence_text: str | None
    source_url: str | None
    confidence: float
    created_at: str = field(default_factory=utc_now)
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReviewItem:
    item_type: str
    field: str
    suggested_value: str | None
    confidence: float
    review_reason: str
    paper_title: str | None = None
    dataset_name: str | None = None
    evidence: str | None = None
    source_url: str | None = None


@dataclass(frozen=True)
class WirelessClassification:
    paper_id: int
    label: Literal["yes", "no", "maybe"]
    confidence: float
    evidence: str
    model_version: str = "keyword-rules-v0"


@dataclass(frozen=True)
class DatasetClaim:
    paper_id: int
    dataset_name: str
    relationship_type: Literal["introduced", "reused", "extended", "compared_against", "unclear"]
    confidence: float
    evidence_text: str
    source_url: str | None = None


@dataclass(frozen=True)
class AvailabilityClaim:
    dataset_id: int | None
    url: str
    availability_status: str
    confidence: float
    evidence_text: str
    checked_at: str = field(default_factory=utc_now)


@dataclass(frozen=True)
class DatasetIdentityDecision:
    dataset_name: str
    canonical_dataset_id: int | None
    decision: Literal["create", "merge", "review"]
    confidence: float
    evidence: str
