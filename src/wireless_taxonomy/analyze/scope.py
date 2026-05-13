from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


WIRELESS_SCOPE_TERMS = {
    "5g",
    "6g",
    "lte",
    "cellular",
    "ran",
    "fronthaul",
    "wi-fi",
    "wifi",
    "802.11",
    "wireless",
    "rf",
    "mimo",
    "antenna",
    "spectrum",
    "mmwave",
    "backscatter",
    "bluetooth",
    "lorawan",
    "satellite",
    "leo",
    "iot",
}

NETWORKING_SCOPE_TERMS = WIRELESS_SCOPE_TERMS | {
    "network",
    "internet",
    "routing",
    "router",
    "congestion",
    "traffic",
    "packet",
    "switch",
    "dataplane",
    "data plane",
    "transport",
    "tcp",
    "rdma",
    "dns",
    "cdn",
    "wan",
    "lan",
    "datacenter",
    "data center",
    "cloud",
    "edge",
    "nic",
    "smartnic",
    "dpu",
    "load balancing",
    "qoe",
    "video streaming",
    "submarine cable",
}

TITLE_SENTENCE_START = re.compile(r"(?i)^(this paper|in this paper|we present|we propose|we introduce|our work)\b")
AUTHOR_SENTENCE_PHRASES = re.compile(r"(?i)\b(this paper|we present|we propose|evaluations? show|we introduce|we evaluate)\b")


@dataclass(frozen=True)
class ScopeIssue:
    paper_id: int | None
    paper_title: str | None
    field: str
    message: str
    evidence: str | None
    confidence: float


@dataclass(frozen=True)
class ScopeAssessment:
    paper_count: int
    networking_like_count: int
    wireless_like_count: int
    malformed_count: int
    networking_like_ratio: float
    wireless_like_ratio: float
    should_proceed: bool
    decision: str
    confidence: float
    issues: list[ScopeIssue]

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_count": self.paper_count,
            "networking_like_count": self.networking_like_count,
            "wireless_like_count": self.wireless_like_count,
            "malformed_count": self.malformed_count,
            "networking_like_ratio": self.networking_like_ratio,
            "wireless_like_ratio": self.wireless_like_ratio,
            "should_proceed": self.should_proceed,
            "decision": self.decision,
            "confidence": self.confidence,
            "issues": [issue.__dict__ for issue in self.issues],
        }


class ScopeAssessor:
    provider_name = "scope_rules_v0"

    def assess(self, papers: list[dict[str, Any]]) -> ScopeAssessment:
        issues: list[ScopeIssue] = []
        networking_like_count = 0
        wireless_like_count = 0
        malformed_count = 0
        for paper in papers:
            text = f"{paper.get('title') or ''} {paper.get('abstract') or ''}".lower()
            if _matches_any(text, NETWORKING_SCOPE_TERMS):
                networking_like_count += 1
            if _matches_any(text, WIRELESS_SCOPE_TERMS):
                wireless_like_count += 1
            malformed_reasons = _malformed_reasons(paper)
            if malformed_reasons:
                malformed_count += 1
                issues.append(
                    ScopeIssue(
                        paper_id=paper.get("id"),
                        paper_title=paper.get("title"),
                        field="Paper Structure",
                        message="; ".join(malformed_reasons),
                        evidence=_structure_evidence(paper),
                        confidence=0.80,
                    )
                )

        paper_count = len(papers)
        denominator = max(paper_count, 1)
        networking_ratio = round(networking_like_count / denominator, 4)
        wireless_ratio = round(wireless_like_count / denominator, 4)
        malformed_ratio = malformed_count / denominator
        should_proceed = networking_ratio >= 0.25 and malformed_ratio <= 0.05 and paper_count > 0
        if not papers:
            decision = "empty_source"
            confidence = 0.95
        elif networking_ratio < 0.25:
            decision = "likely_out_of_scope"
            confidence = 0.85
            issues.append(
                ScopeIssue(
                    paper_id=None,
                    paper_title=None,
                    field="Source Scope",
                    message="Few papers contain networking or wireless scope terms",
                    evidence=f"networking_like_ratio={networking_ratio}",
                    confidence=0.85,
                )
            )
        elif malformed_ratio > 0.05:
            decision = "paper_list_structure_needs_review"
            confidence = 0.80
        elif wireless_ratio < 0.10:
            decision = "networking_source_wireless_sparse"
            confidence = 0.75
        else:
            decision = "in_scope"
            confidence = 0.90
        return ScopeAssessment(
            paper_count=paper_count,
            networking_like_count=networking_like_count,
            wireless_like_count=wireless_like_count,
            malformed_count=malformed_count,
            networking_like_ratio=networking_ratio,
            wireless_like_ratio=wireless_ratio,
            should_proceed=should_proceed,
            decision=decision,
            confidence=confidence,
            issues=issues,
        )


def _matches_any(text: str, terms: set[str]) -> bool:
    normalized = re.sub(r"[^a-z0-9.+-]+", " ", text.lower())
    return any(_contains_term(normalized, term) for term in terms)


def _contains_term(text: str, term: str) -> bool:
    escaped = re.escape(term.lower()).replace(r"\ ", r"\s+")
    return bool(re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", text))


def _malformed_reasons(paper: dict[str, Any]) -> list[str]:
    title = str(paper.get("title") or "").strip()
    authors = str(paper.get("authors") or "").strip()
    reasons: list[str] = []
    if len(title) > 240:
        reasons.append("Paper title is unusually long")
    if len(title) > 140 and TITLE_SENTENCE_START.search(title):
        reasons.append("Paper title looks like abstract prose")
    if len(authors) > 500 and AUTHOR_SENTENCE_PHRASES.search(authors):
        reasons.append("Authors field appears to contain abstract or neighboring paper text")
    return reasons


def _structure_evidence(paper: dict[str, Any]) -> str:
    title = str(paper.get("title") or "")
    authors = str(paper.get("authors") or "")
    return f"title={title[:300]} authors={authors[:300]}"
