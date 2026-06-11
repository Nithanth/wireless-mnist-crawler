from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

_WS_RE = re.compile(r"[^a-z0-9]+")


def normalize_title(title: str | None) -> str:
    """Lowercase, strip punctuation/whitespace -> a stable lookup key."""
    return _WS_RE.sub(" ", (title or "").lower()).strip()


def _doi_key(doi: str | None) -> str:
    return f"doi:{doi.strip().lower()}" if doi and doi.strip() else ""


def _title_key(title: str | None) -> str:
    norm = normalize_title(title)
    return f"title:{norm}" if norm else ""


class MetadataCache:
    """Persistent on-disk index of resolved abstracts and DOIs.

    Once a paper's abstract (or backfilled DOI) is fetched from the network it's
    written here keyed by DOI and by normalized title, so re-runs read from disk
    instead of re-hitting the metadata APIs. This makes the tool fast and
    deterministic to re-run: a paper seen in any previous run is never fetched
    again. The store is a single JSON file::

        {
          "abstracts": {"<key>": {"abstract": ..., "provider": ..., "source_url": ...}},
          "dois":      {"<title-key>": {"doi": ..., "provider": ..., "source_url": ...}},
          "llm":       {"<content+model-hash>": {"label": ..., "confidence": ..., "evidence": ..., "model_version": ...}}
        }

    Keys are ``doi:<doi>`` (preferred) or ``title:<normalized-title>``; LLM labels
    are keyed by a hash of the exact prompt (title+abstract) and model identity,
    so a re-run reuses the saved label unless the inputs or model changed.
    """

    def __init__(self, path: str | os.PathLike[str] | None) -> None:
        self.path = Path(path) if path else None
        self.abstracts: dict[str, dict[str, str]] = {}
        self.dois: dict[str, dict[str, str]] = {}
        self.llm: dict[str, dict[str, Any]] = {}
        self.oa: dict[str, dict[str, Any]] = {}
        self.dirty = False
        if self.path is not None and self.path.exists():
            self._load()

    def _load(self) -> None:
        assert self.path is not None
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        if isinstance(data, dict):
            abstracts = data.get("abstracts")
            dois = data.get("dois")
            llm = data.get("llm")
            oa = data.get("oa")
            if isinstance(abstracts, dict):
                self.abstracts = {k: v for k, v in abstracts.items() if isinstance(v, dict)}
            if isinstance(dois, dict):
                self.dois = {k: v for k, v in dois.items() if isinstance(v, dict)}
            if isinstance(llm, dict):
                self.llm = {k: v for k, v in llm.items() if isinstance(v, dict)}
            if isinstance(oa, dict):
                self.oa = {k: v for k, v in oa.items() if isinstance(v, dict)}

    # -- abstracts -----------------------------------------------------------

    def get_abstract(self, title: str | None, doi: str | None) -> dict[str, str] | None:
        for key in (_doi_key(doi), _title_key(title)):
            if key and key in self.abstracts:
                return self.abstracts[key]
        return None

    def set_abstract(self, title: str | None, doi: str | None, value: dict[str, str]) -> None:
        wrote = False
        for key in (_doi_key(doi), _title_key(title)):
            if key:
                self.abstracts[key] = value
                wrote = True
        if wrote:
            self.dirty = True

    # -- DOIs ----------------------------------------------------------------

    def get_doi(self, title: str | None) -> dict[str, str] | None:
        key = _title_key(title)
        return self.dois.get(key) if key else None

    def set_doi(self, title: str | None, value: dict[str, str]) -> None:
        key = _title_key(title)
        if key:
            self.dois[key] = value
            self.dirty = True

    # -- open-access availability --------------------------------------------

    def get_oa(self, title: str | None, doi: str | None) -> dict[str, Any] | None:
        for key in (_doi_key(doi), _title_key(title)):
            if key and key in self.oa:
                return self.oa[key]
        return None

    def set_oa(self, title: str | None, doi: str | None, value: dict[str, Any]) -> None:
        wrote = False
        for key in (_doi_key(doi), _title_key(title)):
            if key:
                self.oa[key] = value
                wrote = True
        if wrote:
            self.dirty = True

    # -- LLM labels ----------------------------------------------------------

    def get_llm(self, key: str) -> dict[str, Any] | None:
        return self.llm.get(key) if key else None

    def set_llm(self, key: str, value: dict[str, Any]) -> None:
        if key:
            self.llm[key] = value
            self.dirty = True

    # -- persistence ---------------------------------------------------------

    def save(self) -> None:
        """Atomically write the cache to disk if it changed."""
        if self.path is None or not self.dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {"abstracts": self.abstracts, "dois": self.dois, "llm": self.llm, "oa": self.oa}
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        self.dirty = False

    def stats(self) -> dict[str, int]:
        return {"abstracts": len(self.abstracts), "dois": len(self.dois), "llm": len(self.llm), "oa": len(self.oa)}
