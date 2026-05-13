from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from wireless_taxonomy.analyze.datasets import DatasetExtractor
from wireless_taxonomy.analyze.modality_osi import ModalityOsiMapper
from wireless_taxonomy.analyze.wireless import WirelessClassifier
from wireless_taxonomy.config import LlmSettings
from wireless_taxonomy.llm import LlmRequest, LlmRouter


@dataclass(frozen=True)
class AgenticDatasetClaim:
    dataset_name: str
    relationship_type: Literal["introduced", "reused", "extended", "compared_against", "unclear"]
    confidence: float
    evidence_text: str
    source_url: str | None
    modalities: list[str]
    osi_layers: list[str]
    availability_status: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "relationship_type": self.relationship_type,
            "confidence": self.confidence,
            "evidence_text": self.evidence_text,
            "source_url": self.source_url,
            "modalities": self.modalities,
            "osi_layers": self.osi_layers,
            "availability_status": self.availability_status,
        }


@dataclass(frozen=True)
class AgenticPaperAnalysis:
    paper_id: int
    provider_name: str
    wireless_label: Literal["yes", "no", "maybe"]
    is_wireless: bool | None
    wireless_confidence: float
    wireless_evidence: str
    modalities: list[str]
    osi_layers: list[str]
    dataset_claims: list[AgenticDatasetClaim]
    summary: str
    review_needed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "provider_name": self.provider_name,
            "wireless_label": self.wireless_label,
            "is_wireless": self.is_wireless,
            "wireless_confidence": self.wireless_confidence,
            "wireless_evidence": self.wireless_evidence,
            "modalities": self.modalities,
            "osi_layers": self.osi_layers,
            "dataset_claims": [claim.to_dict() for claim in self.dataset_claims],
            "summary": self.summary,
            "review_needed": self.review_needed,
        }


class DeterministicPaperAnalyzer:
    """Stable local analyzer that defines the DB contract before LLM execution."""

    provider_name = "deterministic_agentic_paper_v0"

    def __init__(self):
        self.wireless_classifier = WirelessClassifier()
        self.dataset_extractor = DatasetExtractor()
        self.modality_mapper = ModalityOsiMapper()

    def analyze(self, paper: dict[str, Any], text: str) -> AgenticPaperAnalysis:
        paper_id = int(paper["id"])
        title = str(paper.get("title") or "")
        abstract = str(paper.get("abstract") or "")
        analysis_text = text or "\n\n".join(part for part in [title, abstract] if part)

        wireless = self.wireless_classifier.classify(paper_id, title, analysis_text)
        modalities, osi_layers, modality_confidence = self.modality_mapper.map_text(analysis_text)
        raw_claims = self.dataset_extractor.extract(paper_id, analysis_text, paper.get("paper_url"))
        dataset_claims: list[AgenticDatasetClaim] = []
        for claim in raw_claims:
            claim_confidence = claim.confidence
            if modalities and osi_layers and claim_confidence < 0.92:
                claim_confidence = 0.92
            dataset_claims.append(
                AgenticDatasetClaim(
                    dataset_name=claim.dataset_name,
                    relationship_type=_relationship_type(analysis_text, claim.dataset_name, claim.relationship_type),
                    confidence=claim_confidence,
                    evidence_text=claim.evidence_text,
                    source_url=claim.source_url,
                    modalities=modalities,
                    osi_layers=osi_layers,
                    availability_status=None,
                )
            )

        is_wireless = None
        if wireless.label == "yes" and wireless.confidence >= 0.90:
            is_wireless = True
        elif wireless.label == "no" and wireless.confidence >= 0.90:
            is_wireless = False
        review_needed = (
            wireless.confidence < 0.90
            or bool(dataset_claims and modality_confidence < 0.90)
            or any(claim.confidence < 0.90 for claim in dataset_claims)
        )
        summary = _summary(wireless.label, dataset_claims, modalities, osi_layers)
        return AgenticPaperAnalysis(
            paper_id=paper_id,
            provider_name=self.provider_name,
            wireless_label=wireless.label,
            is_wireless=is_wireless,
            wireless_confidence=wireless.confidence,
            wireless_evidence=wireless.evidence,
            modalities=modalities,
            osi_layers=osi_layers,
            dataset_claims=dataset_claims,
            summary=summary,
            review_needed=review_needed,
        )


class LlmPaperAnalyzer:
    provider_name = "llm_agentic_paper_v0"

    def __init__(self, settings: LlmSettings, router: LlmRouter | None = None):
        self.settings = settings
        self.router = router or LlmRouter(settings)

    def analyze(self, paper: dict[str, Any], text: str) -> AgenticPaperAnalysis:
        paper_id = int(paper["id"])
        response = self.router.complete(
            LlmRequest(
                task="agentic_paper_analysis",
                schema_name="AgenticPaperAnalysis",
                prompt=_llm_prompt(paper, text),
                metadata={"paper_id": paper_id, "title": paper.get("title")},
            )
        )
        if not isinstance(response.parsed, dict):
            raise RuntimeError("LLM paper analysis did not return a JSON object")
        return _analysis_from_payload(
            paper_id=paper_id,
            provider_name=f"{self.provider_name}:{response.provider}:{response.model}",
            payload=response.parsed,
            default_source_url=paper.get("paper_url"),
        )


def _relationship_type(text: str, dataset_name: str, fallback: str) -> Literal["introduced", "reused", "extended", "compared_against", "unclear"]:
    lower = text.lower()
    name = dataset_name.lower()
    if any(phrase in lower for phrase in [f"introduce {name}", f"introduces {name}", f"dataset named {name}", f"dataset called {name}"]):
        return "introduced"
    if any(phrase in lower for phrase in [f"use {name}", f"uses {name}", f"using {name}", f"evaluate on {name}"]):
        return "reused"
    if fallback in {"introduced", "reused", "extended", "compared_against"}:
        return fallback  # type: ignore[return-value]
    return "unclear"


def _summary(label: str, claims: list[AgenticDatasetClaim], modalities: list[str], osi_layers: list[str]) -> str:
    dataset_part = f"{len(claims)} dataset claim(s)" if claims else "no dataset claims"
    modality_part = ", ".join(modalities) if modalities else "no modalities detected"
    osi_part = ", ".join(osi_layers) if osi_layers else "no OSI layers detected"
    return f"wireless={label}; {dataset_part}; modalities={modality_part}; osi={osi_part}"


def _analysis_from_payload(
    paper_id: int,
    provider_name: str,
    payload: dict[str, Any],
    default_source_url: str | None,
) -> AgenticPaperAnalysis:
    wireless = payload.get("wireless") if isinstance(payload.get("wireless"), dict) else {}
    label = _label(wireless.get("label") or payload.get("wireless_label"))
    confidence = _float(wireless.get("confidence") or payload.get("wireless_confidence"), 0.0)
    evidence = _str_or_empty(wireless.get("evidence") or payload.get("wireless_evidence"))
    is_wireless = wireless.get("is_wireless", payload.get("is_wireless"))
    modalities = _str_list(payload.get("modalities"))
    osi_layers = _osi_layers(payload.get("osi_layers"))
    dataset_claims = [
        _dataset_claim_from_payload(item, default_source_url)
        for item in _list(payload.get("datasets") or payload.get("dataset_claims"))
        if isinstance(item, dict)
    ]
    review_needed = bool(payload.get("review_needed")) or confidence < 0.90 or any(claim.confidence < 0.90 for claim in dataset_claims)
    summary = _str_or_empty(payload.get("summary")) or _summary(label, dataset_claims, modalities, osi_layers)
    return AgenticPaperAnalysis(
        paper_id=paper_id,
        provider_name=provider_name,
        wireless_label=label,
        is_wireless=_optional_bool(is_wireless),
        wireless_confidence=confidence,
        wireless_evidence=evidence,
        modalities=modalities,
        osi_layers=osi_layers,
        dataset_claims=dataset_claims,
        summary=summary,
        review_needed=review_needed,
    )


def _dataset_claim_from_payload(payload: dict[str, Any], default_source_url: str | None) -> AgenticDatasetClaim:
    name = _str_or_empty(payload.get("dataset_name") or payload.get("name"))
    if not name:
        raise RuntimeError("LLM dataset claim is missing dataset_name")
    relationship = _relationship(payload.get("relationship_type") or payload.get("relationship"))
    confidence = _float(payload.get("confidence"), 0.0)
    evidence = _str_or_empty(payload.get("evidence_text") or payload.get("evidence"))
    source_url = _optional_str(payload.get("source_url")) or default_source_url
    return AgenticDatasetClaim(
        dataset_name=name,
        relationship_type=relationship,
        confidence=confidence,
        evidence_text=evidence,
        source_url=source_url,
        modalities=_str_list(payload.get("modalities")),
        osi_layers=_osi_layers(payload.get("osi_layers")),
        availability_status=_optional_str(payload.get("availability_status")),
    )


def _llm_prompt(paper: dict[str, Any], text: str) -> str:
    paper_json = json.dumps(
        {
            "id": paper.get("id"),
            "title": paper.get("title"),
            "authors": paper.get("authors"),
            "doi": paper.get("doi"),
            "paper_url": paper.get("paper_url"),
            "abstract": paper.get("abstract"),
        },
        ensure_ascii=False,
    )
    max_chars = 60_000
    return f"""
You analyze one networking/wireless research paper for a taxonomy workbook.

Return JSON only with this shape:
{{
  "wireless": {{
    "label": "yes|no|maybe",
    "is_wireless": true,
    "confidence": 0.0,
    "evidence": "short evidence text"
  }},
  "modalities": ["RF measurements", "SINR"],
  "osi_layers": ["L1", "L4"],
  "datasets": [
    {{
      "dataset_name": "canonical dataset name",
      "relationship_type": "introduced|reused|extended|compared_against|unclear",
      "confidence": 0.0,
      "evidence_text": "short quote or precise paraphrase",
      "source_url": "url if known, otherwise null",
      "modalities": ["RF measurements"],
      "osi_layers": ["L1"],
      "availability_status": "open|closed|unclear|null"
    }}
  ],
  "summary": "concise audit summary",
  "review_needed": false
}}

Rules:
- Only make claims supported by the supplied text.
- Use "maybe" and review_needed=true when evidence is weak.
- Dataset claims must describe datasets/data artifacts used, introduced, extended, or evaluated.
- Do not treat generic words like "data" or "training data" as named datasets unless a dataset name is clear.
- OSI layers must be selected from L1, L2, L3, L4, L5, L6, L7.
- Keep evidence short and tied to the supplied text.

Paper metadata:
{paper_json}

Paper text/snippets:
<<<TEXT
{text[:max_chars]}
TEXT
""".strip()


def _label(value: Any) -> Literal["yes", "no", "maybe"]:
    normalized = str(value or "").strip().lower()
    if normalized in {"yes", "no", "maybe"}:
        return normalized  # type: ignore[return-value]
    return "maybe"


def _relationship(value: Any) -> Literal["introduced", "reused", "extended", "compared_against", "unclear"]:
    normalized = str(value or "").strip().lower()
    if normalized in {"introduced", "reused", "extended", "compared_against", "unclear"}:
        return normalized  # type: ignore[return-value]
    return "unclear"


def _osi_layers(value: Any) -> list[str]:
    allowed = {"L1", "L2", "L3", "L4", "L5", "L6", "L7"}
    return [item for item in _str_list(value) if item in allowed]


def _str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    return None


def _optional_str(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _str_or_empty(value: Any) -> str:
    return str(value).strip() if value is not None else ""
