
import re
import unicodedata


_WS_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9 ]+")
_DOI_PREFIX_RE = re.compile(r"^(https?://(dx\.)?doi\.org/|doi:)", re.IGNORECASE)


def normalize_title(value: str | None) -> str:
    """Canonical form for matching titles across sources.

    Lowercase, strip accents/diacritics, drop punctuation, collapse whitespace.
    """
    if not value:
        return ""
    text = unicodedata.normalize("NFKD", value)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = _NON_ALNUM_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def normalize_doi(value: str | None) -> str:
    """Canonical DOI for matching: lowercase, strip resolver prefixes."""
    if not value:
        return ""
    text = value.strip()
    text = _DOI_PREFIX_RE.sub("", text)
    return text.lower().strip().rstrip("/")
