from __future__ import annotations

from wireless_taxonomy.models import PaperRecord


class MetadataEnricher:
    provider_name = "metadata_passthrough_v0"

    def enrich(self, paper: PaperRecord) -> PaperRecord:
        return paper
