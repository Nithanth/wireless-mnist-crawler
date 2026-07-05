
import json
import os
import re
import tempfile
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

DATASET_USAGE_TTL_DAYS = 30

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def _cache_key_title(title: str | None) -> str:
    """Lowercase, strip punctuation/whitespace -> a stable cache lookup key."""
    return _NON_ALNUM_RE.sub(" ", (title or "").lower()).strip()


def _doi_key(doi: str | None) -> str:
    return f"doi:{doi.strip().lower()}" if doi and doi.strip() else ""


def _title_key(title: str | None) -> str:
    norm = _cache_key_title(title)
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
        self.dataset_usage: dict[str, dict[str, Any]] = {}
        self.dirty = False
        # Guards mutations and save() so the cache is safe to share across
        # worker threads (parallel classify/extract).
        self._lock = threading.RLock()
        if self.path is not None and self.path.exists():
            self._load()

    def _load(self) -> None:
        assert self.path is not None
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            # A corrupt cache silently starting empty would re-spend API
            # credits without the user knowing — warn loudly instead.
            import sys
            print(
                f"WARNING: cache file {self.path} is corrupt or unreadable "
                f"({exc.__class__.__name__}: {exc}). Starting with an empty "
                f"cache — previously cached results will be re-fetched.",
                file=sys.stderr,
            )
            return
        if isinstance(data, dict):
            abstracts = data.get("abstracts")
            dois = data.get("dois")
            llm = data.get("llm")
            oa = data.get("oa")
            du = data.get("dataset_usage")
            if isinstance(abstracts, dict):
                self.abstracts = {k: v for k, v in abstracts.items() if isinstance(v, dict)}
            if isinstance(dois, dict):
                self.dois = {k: v for k, v in dois.items() if isinstance(v, dict)}
            if isinstance(llm, dict):
                self.llm = {k: v for k, v in llm.items() if isinstance(v, dict)}
            if isinstance(oa, dict):
                self.oa = {k: v for k, v in oa.items() if isinstance(v, dict)}
            if isinstance(du, dict):
                self.dataset_usage = {k: v for k, v in du.items() if isinstance(v, dict)}

    # -- abstracts -----------------------------------------------------------

    def get_abstract(self, title: str | None, doi: str | None) -> dict[str, str] | None:
        with self._lock:
            for key in (_doi_key(doi), _title_key(title)):
                if key and key in self.abstracts:
                    return self.abstracts[key]
            return None

    def set_abstract(self, title: str | None, doi: str | None, value: dict[str, str]) -> None:
        with self._lock:
            wrote = False
            for key in (_doi_key(doi), _title_key(title)):
                if key:
                    self.abstracts[key] = value
                    wrote = True
            if wrote:
                self.dirty = True

    # -- DOIs ----------------------------------------------------------------

    def get_doi(self, title: str | None) -> dict[str, str] | None:
        with self._lock:
            key = _title_key(title)
            return self.dois.get(key) if key else None

    def set_doi(self, title: str | None, value: dict[str, str]) -> None:
        with self._lock:
            key = _title_key(title)
            if key:
                self.dois[key] = value
                self.dirty = True

    # -- open-access availability --------------------------------------------

    def get_oa(self, title: str | None, doi: str | None) -> dict[str, Any] | None:
        with self._lock:
            for key in (_doi_key(doi), _title_key(title)):
                if key and key in self.oa:
                    return self.oa[key]
            return None

    def set_oa(self, title: str | None, doi: str | None, value: dict[str, Any]) -> None:
        with self._lock:
            wrote = False
            for key in (_doi_key(doi), _title_key(title)):
                if key:
                    self.oa[key] = value
                    wrote = True
            if wrote:
                self.dirty = True

    # -- dataset usage search ------------------------------------------------

    def get_dataset_usage(self, name: str) -> dict[str, Any] | None:
        with self._lock:
            key = name.strip().lower()
            entry = self.dataset_usage.get(key) if key else None
            if entry is None:
                return None
            stored_at = entry.get("_stored_at")
            if stored_at:
                try:
                    age = datetime.now(timezone.utc) - datetime.fromisoformat(stored_at)
                    if age > timedelta(days=DATASET_USAGE_TTL_DAYS):
                        del self.dataset_usage[key]
                        self.dirty = True
                        return None
                except Exception:
                    pass
            return entry

    def set_dataset_usage(self, name: str, value: dict[str, Any]) -> None:
        with self._lock:
            key = name.strip().lower()
            if key:
                value["_stored_at"] = datetime.now(timezone.utc).isoformat()
                self.dataset_usage[key] = value
                self.dirty = True

    # -- LLM labels ----------------------------------------------------------

    def get_llm(self, key: str) -> dict[str, Any] | None:  # noqa: D401
        with self._lock:
            return self.llm.get(key) if key else None

    def set_llm(self, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            if key:
                self.llm[key] = value
                self.dirty = True

    def delete_llm(self, key: str) -> bool:
        """Thread-safe removal of one LLM entry. Returns True if it existed."""
        with self._lock:
            if key in self.llm:
                del self.llm[key]
                self.dirty = True
                return True
            return False

    def llm_model_census(self) -> dict[str, int]:
        """Count LLM entries grouped by the model_version recorded inside them.

        Entries with no recorded model (legacy) count under ``unrecorded``.
        """
        with self._lock:
            census: dict[str, int] = {}
            for entry in self.llm.values():
                model = str(entry.get("model_version") or "unrecorded")
                census[model] = census.get(model, 0) + 1
            return census

    def gc_llm(self, keep_model: str, drop_unrecorded: bool = False) -> int:
        """Remove LLM entries whose recorded model does NOT match ``keep_model``.

        ``keep_model`` is a case-insensitive substring match against each
        entry's ``model_version`` (e.g. "gemini-3.5-flash" matches both
        "google:gemini-3.5-flash" and "llm_candidate_v0:google:gemini-3.5-flash").
        Legacy entries with no recorded model are kept unless
        ``drop_unrecorded`` is set. Returns the number of entries removed.
        """
        needle = keep_model.strip().lower()
        with self._lock:
            doomed = []
            for key, entry in self.llm.items():
                model = str(entry.get("model_version") or "").lower()
                if not model:
                    if drop_unrecorded:
                        doomed.append(key)
                    continue
                if needle not in model:
                    doomed.append(key)
            for key in doomed:
                del self.llm[key]
            if doomed:
                self.dirty = True
            return len(doomed)

    # -- persistence ---------------------------------------------------------

    def save(self) -> None:
        """Atomically write the cache to disk if it changed."""
        with self._lock:
            if self.path is None or not self.dirty:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload: dict[str, Any] = {"abstracts": self.abstracts, "dois": self.dois, "llm": self.llm, "oa": self.oa, "dataset_usage": self.dataset_usage}
            fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)
                os.replace(tmp, self.path)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
            self.dirty = False

    def clear(self) -> None:
        """Wipe all sections and mark dirty so save() persists the empty state."""
        with self._lock:
            self.abstracts.clear()
            self.dois.clear()
            self.llm.clear()
            self.oa.clear()
            self.dataset_usage.clear()
            self.dirty = True

    def clear_section(self, section: str) -> int:
        """Clear one named section. Returns number of entries removed."""
        with self._lock:
            mapping = {
                "abstracts": self.abstracts,
                "dois": self.dois,
                "llm": self.llm,
                "oa": self.oa,
                "dataset_usage": self.dataset_usage,
            }
            if section not in mapping:
                raise ValueError(f"Unknown section '{section}'. Valid: {list(mapping)}.")
            count = len(mapping[section])
            mapping[section].clear()
            self.dirty = True
            return count

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"abstracts": len(self.abstracts), "dois": len(self.dois), "llm": len(self.llm), "oa": len(self.oa), "dataset_usage": len(self.dataset_usage)}
