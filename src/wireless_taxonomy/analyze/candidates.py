from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from wireless_taxonomy.analyze.wireless import WirelessClassifier
from wireless_taxonomy.config import LlmSettings
from wireless_taxonomy.llm import LlmRequest, LlmRouter

Label = Literal["yes", "no", "maybe"]


@dataclass(frozen=True)
class CandidatePrediction:
    """A single wireless-candidate decision from title + abstract only."""

    paper_id: int
    classifier: str
    model_version: str
    label: Label
    confidence: float
    evidence: str
    used_abstract: bool

    @property
    def high_pass(self) -> bool:
        """Precision-oriented filter: only confident wireless papers."""
        return self.label == "yes"

    @property
    def low_pass(self) -> bool:
        """Recall-oriented filter: keep yes OR maybe candidates."""
        return self.label in {"yes", "maybe"}


class KeywordCandidateClassifier:
    """Deterministic baseline over title + abstract using keyword rules."""

    classifier = "keyword"

    def __init__(self) -> None:
        self._inner = WirelessClassifier()
        self.model_version = self._inner.model_version

    def classify(self, paper: dict[str, Any]) -> CandidatePrediction:
        paper_id = int(paper["id"])
        title = str(paper.get("title") or "")
        abstract = paper.get("abstract")
        result = self._inner.classify(paper_id, title, abstract)
        return CandidatePrediction(
            paper_id=paper_id,
            classifier=self.classifier,
            model_version=self.model_version,
            label=result.label,
            confidence=result.confidence,
            evidence=result.evidence,
            used_abstract=bool(abstract and str(abstract).strip()),
        )


class LlmCandidateClassifier:
    """LLM classifier restricted to title + abstract (no full text)."""

    classifier = "llm"
    provider_name = "llm_candidate_v0"

    def __init__(self, settings: LlmSettings, router: LlmRouter | None = None) -> None:
        self.settings = settings
        self.router = router or LlmRouter(settings)

    def classify(self, paper: dict[str, Any]) -> CandidatePrediction:
        paper_id = int(paper["id"])
        abstract = paper.get("abstract")
        response = self.router.complete(
            LlmRequest(
                task="wireless_candidate_classification",
                schema_name="WirelessCandidate",
                prompt=_prompt(paper),
                metadata={"paper_id": paper_id, "title": paper.get("title")},
            )
        )
        if not isinstance(response.parsed, dict):
            raise RuntimeError("LLM candidate classification did not return a JSON object")
        payload = response.parsed
        return CandidatePrediction(
            paper_id=paper_id,
            classifier=self.classifier,
            model_version=f"{self.provider_name}:{response.provider}:{response.model}",
            label=_label(payload.get("label")),
            confidence=_float(payload.get("confidence"), 0.0),
            evidence=_str(payload.get("evidence")),
            used_abstract=bool(abstract and str(abstract).strip()),
        )


def _prompt(paper: dict[str, Any]) -> str:
    paper_json = json.dumps(
        {
            "title": paper.get("title"),
            "abstract": paper.get("abstract"),
        },
        ensure_ascii=False,
    )
    return f"""
You screen one research paper to decide if it is a WIRELESS / wireless-networking paper.
You only see the title and abstract. Do not assume facts beyond them.

Wireless covers topics such as: cellular (4G/5G/6G/LTE), Wi-Fi/802.11, mmWave, MIMO,
beamforming, RF/spectrum, antennas, channel/CSI/RSSI/SINR measurements, RAN/base stations,
backscatter, LoRa/LPWAN, Bluetooth, satellite/non-terrestrial links, and wireless sensing.
Wired-only networking, pure systems, ML, or theory papers are NOT wireless.

Return JSON only:
{{
  "label": "yes|no|maybe",
  "confidence": 0.0,
  "evidence": "short reason grounded in the title/abstract"
}}

Rules:
- "yes": clearly a wireless paper.
- "no": clearly not wireless.
- "maybe": ambiguous, or the abstract is missing/too thin to be sure.
- Keep evidence to one short sentence.

Paper:
{paper_json}
""".strip()


def _label(value: Any) -> Label:
    normalized = str(value or "").strip().lower()
    if normalized in {"yes", "no", "maybe"}:
        return normalized  # type: ignore[return-value]
    return "maybe"


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _str(value: Any) -> str:
    return str(value).strip() if value is not None else ""
