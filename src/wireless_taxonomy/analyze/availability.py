from __future__ import annotations

from wireless_taxonomy.models import AvailabilityClaim


class AvailabilityChecker:
    provider_name = "availability_rules_v0"

    def check(self, url: str, dataset_id: int | None = None) -> AvailabilityClaim:
        status = "unclear"
        confidence = 0.50
        if any(host in url.lower() for host in ["zenodo", "figshare", "dataverse", "osf.io"]):
            status = "open_downloadable"
            confidence = 0.80
        elif "github.com" in url.lower():
            status = "code_or_data_repository"
            confidence = 0.75
        return AvailabilityClaim(dataset_id, url, status, confidence, f"Rule-based availability classification for {url}")
