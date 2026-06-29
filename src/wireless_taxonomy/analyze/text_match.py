from __future__ import annotations

import re


def title_matches(expected: str | None, actual: str | None) -> bool:
    """Fuzzy title match: exact normalized match or 85%+ token overlap."""
    if not expected or not actual:
        return False
    expected_norm = _normalize(expected)
    actual_norm = _normalize(actual)
    if expected_norm == actual_norm:
        return True
    expected_tokens = set(expected_norm.split())
    actual_tokens = set(actual_norm.split())
    if not expected_tokens or not actual_tokens:
        return False
    overlap = len(expected_tokens & actual_tokens) / max(len(expected_tokens), 1)
    return overlap >= 0.85 and abs(len(expected_tokens) - len(actual_tokens)) <= 4


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value.lower())).strip()
