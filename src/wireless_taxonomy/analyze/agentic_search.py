from __future__ import annotations


class BoundedAgenticSearch:
    provider_name = "agentic_search_disabled_v0"

    def search_dataset_evidence(self, paper_title: str, dataset_name: str | None = None) -> list[dict[str, str]]:
        return [
            {
                "status": "not_run",
                "reason": "Search integration is disabled in this implementation stage.",
                "paper_title": paper_title,
                "dataset_name": dataset_name or "",
            }
        ]
