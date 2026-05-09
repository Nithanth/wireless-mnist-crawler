from __future__ import annotations

import re

from wireless_taxonomy.models import DatasetClaim


class DatasetExtractor:
    provider_name = "dataset_patterns_v0"

    def extract(self, paper_id: int, paper_text: str, source_url: str | None = None) -> list[DatasetClaim]:
        claims: list[DatasetClaim] = []
        for match in re.finditer(r"(?i)(?:dataset|data set)\s+(?:named|called)?\s*([A-Z][A-Za-z0-9 /_-]{2,80})", paper_text):
            name = " ".join(match.group(1).split()).rstrip(".")
            claims.append(DatasetClaim(paper_id, name, "unclear", 0.70, match.group(0), source_url))
        return claims
