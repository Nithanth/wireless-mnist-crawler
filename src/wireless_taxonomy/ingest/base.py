
from abc import ABC, abstractmethod

from wireless_taxonomy.models import PaperSeed, ReviewItem


class IngestAdapter(ABC):
    source_method: str

    @abstractmethod
    def fetch(self) -> list[PaperSeed]:
        raise NotImplementedError


def validate_paper_seeds(seeds: list[PaperSeed], confidence_threshold: float = 0.90) -> list[ReviewItem]:
    review_items: list[ReviewItem] = []
    seen_titles: dict[str, int] = {}
    for seed in seeds:
        normalized_title = " ".join(seed.title.lower().split())
        seen_titles[normalized_title] = seen_titles.get(normalized_title, 0) + 1
        if not seed.title.strip():
            review_items.append(
                ReviewItem("paper", "Paper Title", seed.title, seed.source_confidence, "Missing paper title", source_url=seed.source_url)
            )
        if not seed.authors:
            review_items.append(
                ReviewItem("paper", "Authors", "", seed.source_confidence, "Missing authors", paper_title=seed.title, source_url=seed.source_url)
            )
        if seed.source_confidence < confidence_threshold:
            review_items.append(
                ReviewItem(
                    "paper",
                    "PaperSeed",
                    seed.title,
                    seed.source_confidence,
                    "Source extraction confidence below threshold",
                    paper_title=seed.title,
                    evidence=seed.evidence_text,
                    source_url=seed.source_url,
                )
            )
    for title, count in seen_titles.items():
        if title and count > 1:
            review_items.append(
                ReviewItem("paper", "Paper Title", title, 0.50, "Duplicate paper title detected")
            )
    return review_items
