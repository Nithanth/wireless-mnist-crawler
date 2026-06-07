from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from wireless_taxonomy.analyze.text_match import title_matches

OPENALEX_WORKS_URL = "https://api.openalex.org/works"

# A callable that takes a URL and returns a decoded JSON object. Injected so the
# enricher can be exercised offline in tests.
FetchJson = Callable[[str], dict[str, Any]]


def reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str | None:
    """Rebuild plain-text abstract from OpenAlex's ``abstract_inverted_index``.

    OpenAlex stores abstracts as ``{word: [positions...]}``; we invert that back
    into word order. Returns ``None`` when the index is missing or empty.
    """

    if not inverted_index:
        return None
    positioned: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        if not isinstance(positions, list):
            continue
        for position in positions:
            if isinstance(position, int):
                positioned.append((position, word))
    if not positioned:
        return None
    positioned.sort(key=lambda item: item[0])
    text = " ".join(word for _, word in positioned).strip()
    return text or None


def _clean_doi(doi: str) -> str:
    value = doi.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "http://dx.doi.org/", "doi:"):
        if value.startswith(prefix):
            value = value[len(prefix) :]
    return value


def fetch_work(fetch_json: FetchJson, doi: str | None, title: str | None) -> dict[str, Any] | None:
    """Look up a single OpenAlex work by DOI (exact), then title (verified)."""

    if doi:
        url = f"{OPENALEX_WORKS_URL}/https://doi.org/{quote(_clean_doi(doi), safe='')}"
        try:
            payload = fetch_json(url)
        except Exception:
            payload = {}
        if isinstance(payload, dict) and payload.get("id"):
            return payload
    if title:
        url = OPENALEX_WORKS_URL + "?" + urlencode({"search": title, "per-page": "1"})
        try:
            payload = fetch_json(url)
        except Exception:
            payload = {}
        results = payload.get("results") if isinstance(payload, dict) else None
        if isinstance(results, list) and results and isinstance(results[0], dict):
            candidate = results[0]
            if title_matches(title, candidate.get("title")):
                return candidate
    return None


def abstract_for(fetch_json: FetchJson, doi: str | None, title: str | None) -> str | None:
    """Return the reconstructed abstract for a paper, or ``None`` if unavailable."""

    work = fetch_work(fetch_json, doi, title)
    if not work:
        return None
    return reconstruct_abstract(work.get("abstract_inverted_index"))


def default_fetch_json(url: str) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": "wireless-taxonomy/0.1 (research; mailto:wireless-taxonomy@example.org)"})
    with urlopen(request, timeout=30) as response:  # noqa: S310 - fixed https OpenAlex host
        payload = json.loads(response.read().decode("utf-8"))
    return payload if isinstance(payload, dict) else {}
