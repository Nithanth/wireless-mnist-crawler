from __future__ import annotations

from wireless_taxonomy.models import WirelessClassification


WIRELESS_TERMS = {
    "5g", "6g", "lte", "wi-fi", "wifi", "802.11", "mmwave", "rf", "mimo",
    "antenna", "spectrum", "cellular", "base station", "ran", "csi", "rssi",
    "sinr", "rsrp", "rsrq", "beamforming", "backscatter", "lorawan", "bluetooth",
}


class WirelessClassifier:
    model_version = "keyword-rules-v0"

    def classify(self, paper_id: int, title: str, abstract: str | None = None) -> WirelessClassification:
        text = f"{title} {abstract or ''}".lower()
        matched = sorted(term for term in WIRELESS_TERMS if term in text)
        if matched:
            return WirelessClassification(paper_id, "yes", 0.91, f"Matched wireless terms: {', '.join(matched)}", self.model_version)
        return WirelessClassification(paper_id, "maybe", 0.50, "No strong wireless terms in keyword classifier", self.model_version)
