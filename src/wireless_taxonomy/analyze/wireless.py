from __future__ import annotations

import re

from wireless_taxonomy.models import WirelessClassification


WIRELESS_TERMS = {
    "5g",
    "6g",
    "802.11",
    "antenna",
    "backscatter",
    "base station",
    "beamforming",
    "bluetooth",
    "cellular",
    "channel state information",
    "csi",
    "lora",
    "lorawan",
    "lte",
    "mac layer",
    "mimo",
    "mmwave",
    "ofdm",
    "phy",
    "radio",
    "ran",
    "rf",
    "rssi",
    "rsrp",
    "rsrq",
    "satellite",
    "sinr",
    "spectrum",
    "uwb",
    "wi-fi",
    "wifi",
    "wireless",
    "zigbee",
}

NETWORKING_TERMS = {
    "bandwidth",
    "congestion",
    "datacenter",
    "data center",
    "edge",
    "internet",
    "latency",
    "middlebox",
    "network",
    "packet",
    "routing",
    "sdn",
    "tcp",
    "traffic",
    "transport protocol",
    "wan",
}

COMPUTING_TERMS = {
    "compiler",
    "database",
    "distributed system",
    "file system",
    "gpu",
    "kernel",
    "machine learning",
    "operating system",
    "storage",
}


class WirelessClassifier:
    model_version = "title-abstract-wireless-v1"

    def classify(self, paper_id: int, title: str, abstract: str | None = None) -> WirelessClassification:
        text = _normalize(f"{title} {abstract or ''}")
        wireless = _matched_terms(text, WIRELESS_TERMS)
        networking = _matched_terms(text, NETWORKING_TERMS)
        computing = _matched_terms(text, COMPUTING_TERMS)

        if wireless:
            confidence = min(0.98, 0.91 + (0.02 * min(len(wireless), 3)))
            return WirelessClassification(
                paper_id,
                "yes",
                confidence,
                _evidence("wireless", wireless, networking, computing, abstract),
                self.model_version,
            )

        if networking:
            confidence = 0.82 if abstract else 0.70
            return WirelessClassification(
                paper_id,
                "no",
                confidence,
                _evidence("networking_non_wireless", wireless, networking, computing, abstract),
                self.model_version,
            )

        if computing:
            confidence = 0.76 if abstract else 0.65
            return WirelessClassification(
                paper_id,
                "no",
                confidence,
                _evidence("not_relevant", wireless, networking, computing, abstract),
                self.model_version,
            )

        return WirelessClassification(
            paper_id,
            "maybe",
            0.55 if abstract else 0.45,
            _evidence("uncertain", wireless, networking, computing, abstract),
            self.model_version,
        )


def category_from_evidence(evidence: str | None, label: str | None = None) -> str:
    if evidence:
        match = re.search(r"\bcategory=([a-z_]+)", evidence)
        if match:
            return match.group(1)
    if label == "yes":
        return "wireless"
    if label == "no":
        return "not_relevant"
    return "uncertain"


def _evidence(
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


def _matched_terms(text: str, terms: set[str]) -> list[str]:
    return sorted(term for term in terms if _contains_term(text, term))


def _contains_term(text: str, term: str) -> bool:
    normalized_term = _normalize(term)
    if not normalized_term:
        return False
    if "." in normalized_term:
        return normalized_term in text
    return bool(re.search(rf"\b{re.escape(normalized_term)}\b", text))


def _normalize(value: str) -> str:
    normalized = value.lower().replace("wi fi", "wifi").replace("wi-fi", "wifi")
    normalized = normalized.replace("mm-wave", "mmwave").replace("millimeter wave", "mmwave")
    return re.sub(r"\s+", " ", normalized)
