"""Dataset usage search: find other papers that use a named dataset.

Search order:
1. Semantic Scholar Graph API — abstract keyword search, free, no key needed.
2. GitHub API — repo search for the dataset name, free (30 req/min unauth).
3. LLM web search (Gemini grounding or OpenAI web_search_preview) — fires only
   when S2 returns fewer than MIN_S2_FOR_SKIP results, so we don't waste tokens
   on well-known datasets that S2 already covers well.

Results are cached to avoid re-querying across runs. Returns a count and up to
MAX_USERS paper titles/citations as ``known_users``.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from typing import Any

MAX_USERS = 10
MIN_S2_FOR_SKIP = 3
_UA = "wireless-taxonomy/0.1 (dataset-usage-search)"

_WEB_SEARCH_PROMPT = """Find academic papers and repositories that USE the wireless/networking dataset named "{name}".
I need OTHER papers that reuse this dataset — not the paper that originally introduced it.

Return a JSON object:
{{
  "papers": ["Paper Title (Venue Year)", ...],
  "notes": "one sentence: what this dataset contains and where it is hosted"
}}
List up to 8 papers. Only include results you are confident about from your web search.
If you cannot find any users of this dataset, return {{"papers": [], "notes": ""}}.
"""


def _s2_search(dataset_name: str, limit: int = 20) -> list[str]:
    """Return up to ``limit`` paper titles from Semantic Scholar abstract search."""
    q = urllib.parse.quote(f'"{dataset_name}"')
    url = (
        f"https://api.semanticscholar.org/graph/v1/paper/search"
        f"?query={q}&limit={limit}&fields=title,year,venue"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode())
        papers = data.get("data") or []
        results = []
        for p in papers:
            title = (p.get("title") or "").strip()
            year = p.get("year") or ""
            venue = (p.get("venue") or "").strip()
            if title:
                cite = f"{title} ({venue} {year})".strip(" ()")
                results.append(cite)
        return results[:limit]
    except Exception:
        return []


def _github_search(dataset_name: str, limit: int = 5) -> list[str]:
    """Return up to ``limit`` GitHub repo names mentioning the dataset."""
    q = urllib.parse.quote(dataset_name)
    url = f"https://api.github.com/search/repositories?q={q}&per_page={limit}&sort=stars"
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": _UA, "Accept": "application/vnd.github+json"},
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode())
        items = data.get("items") or []
        return [f"github:{item['full_name']}" for item in items if item.get("full_name")]
    except Exception:
        return []


class DatasetUsageSearcher:
    """Search for papers and repos that use a named dataset.

    Results are cached in the MetadataCache ``dataset_usage`` section so
    repeated runs don't re-query the same dataset names.
    """

    def __init__(self, cache: Any | None = None, use_github: bool = True, router: Any | None = None) -> None:
        self.cache = cache
        self.use_github = use_github
        self.router = router

    def search(self, dataset_name: str) -> dict[str, Any]:
        """Return ``{count, sources, known_users}`` for a dataset name.

        ``known_users`` is a deduplicated list of up to MAX_USERS citations/repos.
        ``sources`` is a dict of ``{provider: count}`` for provenance.
        """
        name = dataset_name.strip()
        if not name:
            return {"count": 0, "sources": {}, "known_users": []}

        if self.cache is not None:
            cached = self.cache.get_dataset_usage(name)
            if cached is not None:
                return cached

        s2_results = _s2_search(name)
        time.sleep(0.4)

        gh_results: list[str] = []
        if self.use_github:
            gh_results = _github_search(name)
            if gh_results:
                time.sleep(0.3)

        web_results: list[str] = []
        web_notes: str = ""
        if self.router is not None and len(s2_results) < MIN_S2_FOR_SKIP:
            web_results, web_notes = _llm_web_search(self.router, name)

        all_users = s2_results + gh_results + web_results
        seen: set[str] = set()
        deduped: list[str] = []
        for u in all_users:
            key = u.lower()
            if key not in seen:
                seen.add(key)
                deduped.append(u)

        sources: dict[str, Any] = {
            "semantic_scholar": len(s2_results),
            "github": len(gh_results),
        }
        if web_results:
            sources["web_search"] = len(web_results)

        result: dict[str, Any] = {
            "count": len(s2_results) + len(gh_results) + len(web_results),
            "sources": sources,
            "known_users": deduped[:MAX_USERS],
        }
        if web_notes:
            result["notes"] = web_notes

        if self.cache is not None:
            self.cache.set_dataset_usage(name, result)

        return result


def _llm_web_search(router: Any, dataset_name: str) -> tuple[list[str], str]:
    """Use the LLM's native web search tool to find papers using this dataset.

    Tries Gemini (Google Search grounding) first since it has the best web
    search integration. Falls back gracefully if no search-capable provider
    is available.
    """
    from wireless_taxonomy.llm import LlmRequest

    prompt = _WEB_SEARCH_PROMPT.format(name=dataset_name)
    try:
        response = router.complete(
            LlmRequest(
                task="dataset_web_search",
                prompt=prompt,
                use_web_search=True,
                metadata={"dataset_name": dataset_name},
            )
        )
        parsed = response.parsed
        if not isinstance(parsed, dict):
            return [], ""
        papers = [str(p).strip() for p in (parsed.get("papers") or []) if str(p).strip()]
        notes = str(parsed.get("notes") or "").strip()
        return papers[:8], notes
    except Exception:
        return [], ""
