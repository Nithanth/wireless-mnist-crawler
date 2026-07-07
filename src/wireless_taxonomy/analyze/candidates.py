
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from wireless_taxonomy.config import LlmSettings
from wireless_taxonomy.llm import LlmRequest, LlmRouter

Label = Literal["yes", "no", "maybe"]

WIRELESS_TERMS = {
    "5g", "6g", "802.11", "antenna", "backscatter", "base station", "beamforming",
    "bluetooth", "cellular", "channel state information", "csi", "lora", "lorawan",
    "lte", "mac layer", "mimo", "mmwave", "ofdm", "phy", "radio", "ran", "rf",
    "rssi", "rsrp", "rsrq", "satellite", "sinr", "spectrum", "uwb", "wi-fi", "wifi",
    "wireless", "zigbee",
}

NETWORKING_TERMS = {
    "bandwidth", "congestion", "datacenter", "data center", "edge", "internet",
    "latency", "middlebox", "network", "packet", "routing", "sdn", "tcp", "traffic",
    "transport protocol", "wan",
}

COMPUTING_TERMS = {
    "compiler", "database", "distributed system", "file system", "gpu", "kernel",
    "machine learning", "operating system", "storage",
}


def _normalize(value: str) -> str:
    normalized = value.lower().replace("wi fi", "wifi").replace("wi-fi", "wifi")
    normalized = normalized.replace("mm-wave", "mmwave").replace("millimeter wave", "mmwave")
    return re.sub(r"\s+", " ", normalized)


def _contains_term(text: str, term: str) -> bool:
    normalized_term = _normalize(term)
    if not normalized_term:
        return False
    if "." in normalized_term:
        return normalized_term in text
    return bool(re.search(rf"\b{re.escape(normalized_term)}\b", text))


def _matched_terms(text: str, terms: set[str]) -> list[str]:
    return sorted(term for term in terms if _contains_term(text, term))


def _keyword_evidence(
    category: str,
    wireless: list[str],
    networking: list[str],
    computing: list[str],
    abstract: str | None,
) -> str:
    parts = [f"category={category}"]
    if wireless:
        parts.append(f"wireless_terms={', '.join(wireless)}")
    if networking:
        parts.append(f"networking_terms={', '.join(networking)}")
    if computing:
        parts.append(f"computing_terms={', '.join(computing)}")
    if not abstract:
        parts.append("abstract_missing=true")
    return "; ".join(parts)


@dataclass(frozen=True)
class CandidatePrediction:
    """A single wireless-candidate decision from title + abstract only."""

    paper_id: int
    classifier: str
    model_version: str
    label: Label
    confidence: str  # "high", "medium", or "low"
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
    model_version = "keyword-rules-v1"

    def classify(self, paper: dict[str, Any]) -> CandidatePrediction:
        paper_id = int(paper["id"])
        title = str(paper.get("title") or "")
        abstract = paper.get("abstract")
        text = _normalize(f"{title} {abstract or ''}")
        wireless = _matched_terms(text, WIRELESS_TERMS)
        networking = _matched_terms(text, NETWORKING_TERMS)
        computing = _matched_terms(text, COMPUTING_TERMS)

        if wireless:
            label: Label = "yes"
            confidence = "high"
            category = "wireless"
        elif networking:
            label = "no"
            confidence = "high" if abstract else "medium"
            category = "networking_non_wireless"
        elif computing:
            label = "no"
            confidence = "medium"
            category = "not_relevant"
        else:
            label = "maybe"
            confidence = "low" if not abstract else "medium"
            category = "uncertain"

        return CandidatePrediction(
            paper_id=paper_id,
            classifier=self.classifier,
            model_version=self.model_version,
            label=label,
            confidence=confidence,
            evidence=_keyword_evidence(category, wireless, networking, computing, abstract),
            used_abstract=bool(abstract and str(abstract).strip()),
        )


class LlmCandidateClassifier:
    """LLM classifier using title + abstract, or full PDF when available."""

    classifier = "llm"
    provider_name = "llm_candidate_v0"

    def __init__(
        self,
        settings: LlmSettings,
        router: LlmRouter | None = None,
        cache: Any | None = None,
        refresh: bool = False,
    ) -> None:
        self.settings = settings
        self.router = router or LlmRouter(settings)
        self.cache = cache
        self.refresh = refresh

    def _model_identity(self) -> str:
        """Stable identifier for the primary provider/model only.

        Part of the cache key so a label is reused only while the model that
        produced it is unchanged; swapping the primary model invalidates it.
        Fallbacks are a resilience mechanism and do NOT change the experiment
        identity — adding or removing them should not bust the cache.
        """
        try:
            provider = self.router.select_provider()
            return f"{provider.provider}/{provider.model}"
        except Exception:
            return ""

    def classify(
        self,
        paper: dict[str, Any],
        pdf_bytes: bytes | None = None,
        pdf_text: str | None = None,
        refresh: bool = False,
        classify_no_pdf: bool = False,
    ) -> CandidatePrediction:
        paper_id = int(paper["id"])
        abstract = paper.get("abstract")
        used_abstract = bool(abstract and str(abstract).strip())

        # --classify-no-pdf: use keyword snippets from PDF text instead of
        # sending the full PDF. Much cheaper (~500 tokens vs ~28K) with
        # comparable accuracy for the binary wireless/not decision.
        snippets = ""
        effective_pdf_bytes = pdf_bytes
        if classify_no_pdf:
            effective_pdf_bytes = None
            if pdf_text:
                snippets = extract_keyword_snippets(pdf_text)
            elif pdf_bytes:
                # Extract text from PDF bytes on the fly for snippet search
                try:
                    from wireless_taxonomy.llm import _pdf_bytes_to_text
                    text = _pdf_bytes_to_text(pdf_bytes)
                    if text:
                        snippets = extract_keyword_snippets(text)
                except Exception:
                    pass

        prompt = _prompt(paper, has_pdf=effective_pdf_bytes is not None, keyword_snippets=snippets)
        model_id = self._model_identity()
        cache_key = _llm_cache_key(self.provider_name, model_id, prompt)
        if self.cache is not None and not self.refresh and not refresh:
            cached = self.cache.get_llm(cache_key)
            if cached is None:
                # Migrate from the old full-chain key format (pre-refactor):
                # the old key included every provider with an API key, e.g.
                # "google/gemini-3.5-flash,openai/gpt-5.4-mini,anthropic/...".
                # Reconstruct that chain from ALL configured keys (not the
                # fallback setting, which may have changed) and adopt the
                # legacy entry under the new primary-only key.
                try:
                    legacy_providers = self.router.all_configured_providers()
                    legacy_id = ",".join(f"{p.provider}/{p.model}" for p in legacy_providers)
                    if legacy_id and legacy_id != model_id:
                        legacy_key = _llm_cache_key(self.provider_name, legacy_id, prompt)
                        cached = self.cache.get_llm(legacy_key)
                        if cached is not None:
                            self.cache.set_llm(cache_key, cached)
                except Exception:
                    pass
            if cached is not None:
                return CandidatePrediction(
                    paper_id=paper_id,
                    classifier=self.classifier,
                    model_version=_str(cached.get("model_version")) or self.provider_name,
                    label=_label(cached.get("label")),
                    confidence=_confidence(cached.get("confidence")),
                    evidence=_str(cached.get("evidence")),
                    used_abstract=used_abstract,
                )
        response = self.router.complete(
            LlmRequest(
                task="wireless_candidate_classification",
                schema_name="WirelessCandidate",
                prompt=prompt,
                metadata={"paper_id": paper_id, "title": paper.get("title")},
                pdf_bytes=effective_pdf_bytes,
            )
        )
        if not isinstance(response.parsed, dict):
            raise RuntimeError("LLM candidate classification did not return a JSON object")
        payload = response.parsed
        prediction = CandidatePrediction(
            paper_id=paper_id,
            classifier=self.classifier,
            model_version=f"{self.provider_name}:{response.provider}:{response.model}",
            label=_label(payload.get("label")),
            confidence=_confidence(payload.get("confidence")),
            evidence=_str(payload.get("evidence")),
            used_abstract=used_abstract,
        )
        if self.cache is not None:
            self.cache.set_llm(
                cache_key,
                {
                    "label": prediction.label,
                    "confidence": prediction.confidence,
                    "evidence": prediction.evidence,
                    "model_version": prediction.model_version,
                },
            )
        return prediction


def _llm_cache_key(provider_name: str, model_identity: str, prompt: str) -> str:
    """Content-addressed key over prompt (title+abstract) and model identity.

    The full prompt text is hashed, so any change to the title, abstract, or
    prompt template naturally invalidates the entry; ``model_identity`` ties it
    to the specific model chain that produced the label.
    """
    digest = hashlib.sha256(f"{provider_name}\x00{model_identity}\x00{prompt}".encode()).hexdigest()
    return f"v1:{digest}"


_SNIPPET_KEYWORDS = WIRELESS_TERMS | {
    "dataset", "measurement", "trace", "testbed", "experiment",
    "deployment", "field trial", "data collection", "open source",
}

# Max chars of keyword snippets to include in the classify prompt
_MAX_SNIPPET_CHARS = 3000


def extract_keyword_snippets(pdf_text: str, max_chars: int = _MAX_SNIPPET_CHARS) -> str:
    """Search PDF text for lines containing wireless/data keywords.

    Returns a compact string of relevant snippets (deduplicated, ordered by
    position) that give the classifier extra context without sending the full PDF.
    """
    if not pdf_text or len(pdf_text) < 200:
        return ""
    normalized = _normalize(pdf_text)
    lines = normalized.split("\n")
    # Score each line by keyword hits
    scored: list[tuple[int, int, str]] = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if len(stripped) < 20 or len(stripped) > 500:
            continue
        hits = sum(1 for term in _SNIPPET_KEYWORDS if _contains_term(stripped, term))
        if hits > 0:
            scored.append((hits, idx, stripped))
    # Sort by score (descending), break ties by position
    scored.sort(key=lambda x: (-x[0], x[1]))
    # Collect snippets up to the char budget
    seen: set[str] = set()
    snippets: list[tuple[int, str]] = []
    total = 0
    for _hits, idx, line in scored:
        if line in seen:
            continue
        if total + len(line) > max_chars:
            break
        seen.add(line)
        snippets.append((idx, line))
        total += len(line)
    # Return in document order
    snippets.sort(key=lambda x: x[0])
    return "\n".join(s for _, s in snippets)


def _prompt(paper: dict[str, Any], has_pdf: bool = False, keyword_snippets: str = "") -> str:
    paper_json = json.dumps(
        {
            "title": paper.get("title"),
            "abstract": paper.get("abstract"),
        },
        ensure_ascii=False,
    )
    if keyword_snippets:
        context_note = (
            "You have the title, abstract, and keyword-matched excerpts from the "
            "full paper below. Use all of this to judge."
        )
    elif has_pdf:
        context_note = (
            "You have the full paper PDF attached. Use ALL available content to judge."
        )
    else:
        context_note = (
            "You only see the title and abstract. When information is limited, "
            "lean toward \"maybe\" rather than \"no\" — we prefer to include "
            "borderline papers rather than miss wireless papers."
        )

    snippet_section = ""
    if keyword_snippets:
        snippet_section = (
            "\nKeyword-matched excerpts from the full paper text:\n---\n"
            f"{keyword_snippets}\n---\n"
        )

    return f"""
You screen one research paper to decide if it is a WIRELESS / wireless-networking paper.
{context_note}

Wireless covers (non-exhaustive):
- Cellular: 4G/LTE, 5G/NR, 6G, RAN, base stations, O-RAN, small cells, HetNets
- Wi-Fi / 802.11: any variant (ax, be, ad, ay), WLAN, access points
- Millimeter-wave (mmWave), sub-THz, terahertz communications
- MIMO, massive MIMO, beamforming, antenna design, phased arrays
- RF / spectrum: channel modeling, CSI, RSSI, SINR, propagation, spectrum sharing,
  cognitive radio, dynamic spectrum access
- IoT wireless: LoRa, LPWAN, Zigbee, Bluetooth/BLE, UWB, RFID, NFC, backscatter
- Satellite / non-terrestrial networks: LEO, GEO, direct-to-cell, satellite IoT
- Wireless sensing: radar, WiFi sensing, RF sensing, localization, positioning
- Vehicular: V2X, V2V, DSRC, C-V2X
- Device-to-device (D2D), mesh, ad-hoc networks
- Mobile edge / MEC when the wireless link is central
- ML/AI *applied to* wireless problems (e.g., deep learning for channel estimation,
  RL for spectrum management) — these ARE wireless papers

NOT wireless (label "no"):
- Wired/datacenter networking (switches, RDMA, flow scheduling, optical fiber)
- Pure distributed systems, storage, OS, or databases
- Video/streaming/CDN where the radio link is merely the last-hop access
- Pure ML/theory with no wireless application

Judge by the paper's *central* contribution. Label "yes" when the wireless
link/medium is core to the work, not just incidentally mentioned.

Examples:
- Title: "EdgeRIC: Empowering Real-time Intelligent Optimization and Control in NextG Cellular Networks"
  → {{"label": "yes", "confidence": "high", "evidence": "Core contribution is a RAN intelligent controller for 5G cellular."}}
- Title: "DINT: Fast In-Kernel Distributed Transactions with eBPF"
  → {{"label": "no", "confidence": "high", "evidence": "In-kernel datacenter transactions, no wireless component."}}
- Title: "Habitus: Boosting Mobile Immersive Content Delivery through Full-body Pose Tracking and Multipath Networking"
  → {{"label": "maybe", "confidence": "medium", "evidence": "Uses mmWave multipath but main focus is immersive content delivery."}}

Return JSON only:
{{
  "label": "yes" or "no" or "maybe",
  "confidence": "high" or "medium" or "low",
  "evidence": "short reason grounded in the paper content"
}}

Rules:
- "yes": wireless link/medium is clearly central to the paper.
- "no": clearly not about wireless.
- "maybe": ambiguous, mixed, or not enough information to be sure.
- confidence: "high" = very certain, "medium" = fairly sure, "low" = uncertain.
- Keep evidence to one short sentence.

Paper:
{paper_json}
{snippet_section}""".strip()


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


def _confidence(value: Any) -> str:
    """Parse confidence — accepts categorical or legacy numeric values."""
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("high", "medium", "low"):
            return s
    if isinstance(value, (int, float)):
        if value >= 0.85:
            return "high"
        if value >= 0.60:
            return "medium"
        return "low"
    return "medium"


def _str(value: Any) -> str:
    return str(value).strip() if value is not None else ""
