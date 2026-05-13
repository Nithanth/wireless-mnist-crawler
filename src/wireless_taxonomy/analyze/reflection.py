from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from wireless_taxonomy.analyze.full_text import _normalize_title


@dataclass(frozen=True)
class ReflectionIssue:
    field: str
    reason: str
    suggested_value: str | None
    confidence: float
    evidence: str | None
    source_url: str | None
    dataset_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "reason": self.reason,
            "suggested_value": self.suggested_value,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "source_url": self.source_url,
            "dataset_name": self.dataset_name,
        }


@dataclass(frozen=True)
class PaperReflection:
    paper_id: int
    decision: Literal["accepted", "review"]
    confidence: float
    issues: list[ReflectionIssue]

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "decision": self.decision,
            "confidence": self.confidence,
            "issues": [issue.to_dict() for issue in self.issues],
        }


class DeterministicAnalysisReflector:
    provider_name = "deterministic_reflection_v0"

    def reflect(self, paper: dict[str, Any], analysis: dict[str, Any], dataset_claims: list[dict[str, Any]], text: str) -> PaperReflection:
        issues: list[ReflectionIssue] = []
        paper_id = int(paper["id"])
        wireless_evidence = str(analysis.get("wireless_evidence") or "").strip()
        wireless_label = str(analysis.get("wireless_label") or "").strip()
        if wireless_label in {"yes", "no"} and not wireless_evidence:
            issues.append(
                ReflectionIssue(
                    "Wireless Classification",
                    "Wireless decision is missing evidence text",
                    wireless_label,
                    0.50,
                    None,
                    paper.get("paper_url"),
                )
            )
        elif wireless_evidence and not _evidence_is_grounded(wireless_evidence, text):
            issues.append(
                ReflectionIssue(
                    "Wireless Classification",
                    "Wireless evidence does not appear grounded in available paper text",
                    wireless_label,
                    0.55,
                    wireless_evidence,
                    paper.get("paper_url"),
                )
            )
        for claim in dataset_claims:
            issues.extend(self._dataset_issues(claim, text))
        decision: Literal["accepted", "review"] = "review" if issues else "accepted"
        confidence = 0.95 if not issues else min(issue.confidence for issue in issues)
        return PaperReflection(paper_id, decision, confidence, issues)

    def _dataset_issues(self, claim: dict[str, Any], text: str) -> list[ReflectionIssue]:
        issues: list[ReflectionIssue] = []
        dataset_name = str(claim.get("dataset_name") or "").strip()
        evidence = str(claim.get("evidence_text") or "").strip()
        source_url = claim.get("source_url")
        confidence = float(claim.get("confidence") or 0.0)
        modalities = _json_list(claim.get("modalities_json"))
        osi_layers = _json_list(claim.get("osi_layers_json"))
        if _generic_dataset_name(dataset_name):
            issues.append(
                ReflectionIssue(
                    "Dataset Claim",
                    "Dataset name is generic rather than a named dataset/artifact",
                    dataset_name,
                    0.45,
                    evidence or None,
                    source_url,
                    dataset_name,
                )
            )
        if confidence < 0.90:
            issues.append(
                ReflectionIssue(
                    "Dataset Claim",
                    "Dataset claim confidence is below threshold",
                    dataset_name,
                    confidence,
                    evidence or None,
                    source_url,
                    dataset_name,
                )
            )
        if not evidence:
            issues.append(
                ReflectionIssue(
                    "Dataset Claim",
                    "Dataset claim is missing evidence text",
                    dataset_name,
                    0.50,
                    None,
                    source_url,
                    dataset_name,
                )
            )
        elif not _evidence_is_grounded(evidence, text):
            issues.append(
                ReflectionIssue(
                    "Dataset Claim",
                    "Dataset evidence does not appear grounded in available paper text",
                    dataset_name,
                    0.55,
                    evidence,
                    source_url,
                    dataset_name,
                )
            )
        if not modalities:
            issues.append(
                ReflectionIssue(
                    "Modality",
                    "Dataset claim is missing modality evidence",
                    dataset_name,
                    0.60,
                    evidence or None,
                    source_url,
                    dataset_name,
                )
            )
        if not osi_layers:
            issues.append(
                ReflectionIssue(
                    "OSI Layer",
                    "Dataset claim is missing OSI layer evidence",
                    dataset_name,
                    0.60,
                    evidence or None,
                    source_url,
                    dataset_name,
                )
            )
        return issues


def _evidence_is_grounded(evidence: str, text: str) -> bool:
    if not evidence or not text:
        return False
    normalized_evidence = _normalize_title(evidence)
    normalized_text = _normalize_title(text[:120_000])
    if normalized_evidence and normalized_evidence in normalized_text:
        return True
    evidence_tokens = [token for token in normalized_evidence.split() if len(token) >= 4]
    if len(evidence_tokens) < 4:
        return True
    text_tokens = set(normalized_text.split())
    overlap = sum(1 for token in evidence_tokens if token in text_tokens) / len(evidence_tokens)
    return overlap >= 0.55


def _generic_dataset_name(value: str) -> bool:
    normalized = _normalize_title(value)
    return normalized in {"data", "dataset", "datasets", "training data", "test data", "our data", "the dataset", "benchmark"}


def _json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if not isinstance(value, str):
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if str(item).strip()]
