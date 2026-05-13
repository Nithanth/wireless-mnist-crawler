from __future__ import annotations

import re
from urllib.parse import urlparse


def title_matches(expected: str | None, actual: str | None) -> bool:
    if not expected or not actual:
        return False
    expected_norm = normalize_title(expected)
    actual_norm = normalize_title(actual)
    if expected_norm == actual_norm:
        return True
    expected_tokens = set(expected_norm.split())
    actual_tokens = set(actual_norm.split())
    if not expected_tokens or not actual_tokens:
        return False
    overlap = len(expected_tokens & actual_tokens) / max(len(expected_tokens), 1)
    return overlap >= 0.85 and abs(len(expected_tokens) - len(actual_tokens)) <= 4


def text_matches_title(expected_title: str | None, text: str) -> bool:
    if not expected_title or not text:
        return False
    head = text[:5000]
    if normalize_title(expected_title) in normalize_title(head):
        return True
    tokens = significant_title_tokens(expected_title)
    if not tokens:
        return False
    normalized_head_tokens = set(normalize_title(head).split())
    required = min(len(tokens), 8)
    matched = sum(1 for token in tokens[:12] if token in normalized_head_tokens)
    return matched >= max(3, int(required * 0.75))


def significant_title_tokens(title: str) -> list[str]:
    stopwords = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "into",
        "using",
        "towards",
        "toward",
        "this",
        "that",
        "paper",
        "a",
        "an",
        "of",
        "to",
        "in",
        "on",
        "at",
        "by",
        "as",
    }
    tokens = re.findall(r"[a-zA-Z0-9]+", title.lower())
    return [token for token in tokens if len(token) >= 3 and token not in stopwords]


def normalize_title(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value.lower())).strip()


def normalize_authors(value: str | None) -> str:
    return "|".join(author_names(value)[:10])


def author_names(value: str | None) -> list[str]:
    if not value:
        return []
    pieces = re.split(r"\s*(?:,|;|\band\b)\s*", value)
    names: list[str] = []
    for piece in pieces:
        normalized = normalize_person_name(piece)
        if normalized:
            names.append(normalized)
    return unique(names)


def author_overlap_score(expected: list[str], actual: list[str]) -> float:
    if not expected or not actual:
        return 0.0
    expected_last_names = {last_name(name) for name in expected if last_name(name)}
    actual_last_names = {last_name(name) for name in actual if last_name(name)}
    if not expected_last_names or not actual_last_names:
        return 0.0
    overlap = expected_last_names & actual_last_names
    if len(overlap) >= 2:
        return 0.20
    if len(overlap) == 1:
        return 0.12
    return 0.0


def normalize_person_name(value: str) -> str:
    normalized = re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z\s\-']", " ", value)).strip().lower()
    return normalized if len(normalized) >= 2 else ""


def last_name(name: str) -> str:
    parts = [part for part in re.split(r"\s+", name.strip()) if part]
    return parts[-1] if parts else ""


def same_doi(left: str, right: str) -> bool:
    return normalize_doi(left) == normalize_doi(right)


def normalize_doi(value: str | None) -> str:
    if not value:
        return ""
    normalized = value.strip().lower()
    normalized = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", normalized)
    normalized = normalized.removeprefix("doi:")
    return normalized.strip()


def resolver_key(doi: str | None, title: str | None) -> str:
    normalized_doi = normalize_doi(doi)
    normalized_title = normalize_title(title or "")
    return f"doi:{normalized_doi}|title:{normalized_title}"


def unique(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = candidate_key(value)
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def candidate_key(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.rstrip("/")
    if host == "arxiv.org" and (path.startswith("/pdf/") or path.startswith("/abs/")):
        arxiv_id = path.rsplit("/", 1)[-1]
        arxiv_id = re.sub(r"v\d+$", "", arxiv_id)
        prefix = "/pdf/" if path.startswith("/pdf/") else "/abs/"
        return f"{host}{prefix}{arxiv_id}".lower()
    return url.strip().lower()
