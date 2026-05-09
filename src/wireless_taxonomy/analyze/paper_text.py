from __future__ import annotations


class PaperTextFetcher:
    provider_name = "paper_text_local_v0"

    def fetch_text(self, title: str, abstract: str | None = None, pdf_url: str | None = None) -> str:
        return "\n\n".join(part for part in [title, abstract or "", f"PDF: {pdf_url}" if pdf_url else ""] if part)
