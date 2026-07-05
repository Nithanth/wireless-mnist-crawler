"""Corpus management: auto-named, versioned, isolated corpus directories.

A *corpus* is a self-contained unit of work: one SQLite DB, one results
directory, and metadata recording which model built it.  Corpora live under
``corpora/<name>/`` and are auto-named ``corpus_v1``, ``corpus_v2``, ... unless
the user overrides with an explicit name.

Layout::

    corpora/
      ACTIVE                     # single line: name of the active corpus
      corpus_v1/
        taxonomy.sqlite          # corpus DB (papers, claims, datasets)
        results/                 # per-venue CSVs + master files
        snapshots/               # timestamped DB copies for rollback
        corpus.json              # metadata (model identity, history)

Model-change policy: each corpus records the primary LLM identity that built
it.  Running with a different model against an existing corpus produces a
loud warning (mixing models pollutes comparability) and the recommended path
is a new corpus version.

Legacy compatibility: if ``corpora/`` does not exist, the repo-root layout
(``taxonomy.sqlite`` + ``src/results/``) keeps working exactly as before.
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

CORPORA_DIR = Path("corpora")
ACTIVE_FILE = "ACTIVE"
META_FILE = "corpus.json"
DB_FILE = "taxonomy.sqlite"
RESULTS_DIR = "results"
SNAPSHOTS_DIR = "snapshots"
MAX_SNAPSHOTS = 5

_AUTO_NAME_RE = re.compile(r"^corpus_v(\d+)$")


class Corpus:
    """Handle to one corpus directory with path helpers and metadata."""

    def __init__(self, name: str, root: Path = CORPORA_DIR) -> None:
        self.name = name
        self.dir = root / name

    @property
    def db_path(self) -> Path:
        return self.dir / DB_FILE

    @property
    def results_dir(self) -> Path:
        return self.dir / RESULTS_DIR

    @property
    def snapshots_dir(self) -> Path:
        return self.dir / SNAPSHOTS_DIR

    @property
    def meta_path(self) -> Path:
        return self.dir / META_FILE

    def exists(self) -> bool:
        return self.dir.is_dir()

    # ── metadata ─────────────────────────────────────────────────────────────

    def read_meta(self) -> dict:
        if self.meta_path.exists():
            try:
                return json.loads(self.meta_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def write_meta(self, meta: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def record_run(self, model_identity: str, venues_years: str) -> None:
        """Append a run record and set the model identity if unset."""
        meta = self.read_meta()
        meta.setdefault("name", self.name)
        meta.setdefault("created_at", _now())
        meta.setdefault("model_identity", model_identity)
        meta.setdefault("runs", []).append({
            "at": _now(),
            "model_identity": model_identity,
            "venues_years": venues_years,
        })
        self.write_meta(meta)

    def model_identity(self) -> str:
        return str(self.read_meta().get("model_identity") or "")

    # ── snapshots / rollback ─────────────────────────────────────────────────

    def snapshot(self) -> Path | None:
        """Copy the DB into snapshots/ with a timestamp. Returns the path,
        or None if there is no DB yet.  Keeps at most MAX_SNAPSHOTS."""
        if not self.db_path.exists():
            return None
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        dest = self.snapshots_dir / f"{stamp}.sqlite"
        # Never overwrite an existing snapshot (e.g. two snapshots within the
        # same second during a rollback) — add a numeric suffix instead.
        n = 1
        while dest.exists():
            dest = self.snapshots_dir / f"{stamp}_{n}.sqlite"
            n += 1
        shutil.copy2(self.db_path, dest)
        # Prune oldest beyond MAX_SNAPSHOTS
        snaps = sorted(self.snapshots_dir.glob("*.sqlite"))
        for old in snaps[:-MAX_SNAPSHOTS]:
            old.unlink(missing_ok=True)
        return dest

    def list_snapshots(self) -> list[Path]:
        if not self.snapshots_dir.is_dir():
            return []
        return sorted(self.snapshots_dir.glob("*.sqlite"))

    def rollback(self, snapshot_name: str) -> Path:
        """Restore the DB from a snapshot (by filename or timestamp prefix).

        The current DB is itself snapshotted first so a rollback is always
        reversible.  Raises FileNotFoundError if no matching snapshot.
        """
        snaps = self.list_snapshots()
        matches = [s for s in snaps if s.name == snapshot_name or s.stem == snapshot_name]
        if not matches:
            matches = [s for s in snaps if s.stem.startswith(snapshot_name)]
        if not matches:
            raise FileNotFoundError(
                f"No snapshot matching '{snapshot_name}' in {self.snapshots_dir}"
            )
        source = matches[-1]
        self.snapshot()  # preserve current state before overwriting
        shutil.copy2(source, self.db_path)
        return source


# ── module-level helpers ──────────────────────────────────────────────────────


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def list_corpora(root: Path = CORPORA_DIR) -> list[Corpus]:
    if not root.is_dir():
        return []
    return [
        Corpus(p.name, root) for p in sorted(root.iterdir())
        if p.is_dir() and (p / META_FILE).exists()
    ]


def next_auto_name(root: Path = CORPORA_DIR) -> str:
    """Next free auto-name: corpus_v1, corpus_v2, ..."""
    versions = [
        int(m.group(1))
        for c in list_corpora(root)
        if (m := _AUTO_NAME_RE.match(c.name))
    ]
    return f"corpus_v{max(versions, default=0) + 1}"


def active_corpus(root: Path = CORPORA_DIR) -> Corpus | None:
    """The corpus named in corpora/ACTIVE, if it exists."""
    active_path = root / ACTIVE_FILE
    if active_path.exists():
        name = active_path.read_text(encoding="utf-8").strip()
        if name:
            c = Corpus(name, root)
            if c.exists():
                return c
    return None


def set_active(name: str, root: Path = CORPORA_DIR) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / ACTIVE_FILE).write_text(name + "\n", encoding="utf-8")


def resolve_corpus(
    name: str | None = None,
    create: bool = True,
    root: Path = CORPORA_DIR,
) -> Corpus:
    """Resolve the corpus to operate on.

    Priority: explicit name > active corpus > auto-create corpus_vN.
    When a new corpus is created it becomes the active one.
    """
    if name:
        c = Corpus(name, root)
        if not c.exists() and create:
            c.dir.mkdir(parents=True, exist_ok=True)
            c.results_dir.mkdir(parents=True, exist_ok=True)
            c.write_meta({"name": name, "created_at": _now()})
            set_active(name, root)
        return c
    existing = active_corpus(root)
    if existing is not None:
        return existing
    if not create:
        raise FileNotFoundError("No active corpus and no name given.")
    auto = next_auto_name(root)
    c = Corpus(auto, root)
    c.dir.mkdir(parents=True, exist_ok=True)
    c.results_dir.mkdir(parents=True, exist_ok=True)
    c.write_meta({"name": auto, "created_at": _now()})
    set_active(auto, root)
    return c


def check_model_compatibility(corpus: Corpus, current_model: str) -> str | None:
    """Return a warning message if the current model differs from the one
    that built this corpus, else None.  Empty recorded identity means the
    corpus predates model tracking — no warning, it gets recorded on the
    next run.
    """
    recorded = corpus.model_identity()
    if recorded and current_model and recorded != current_model:
        return (
            f"Corpus '{corpus.name}' was built with model '{recorded}' but the "
            f"current primary model is '{current_model}'. Mixing models in one "
            f"corpus hurts comparability. Recommended: start a new corpus "
            f"(next auto-name: {next_auto_name()}). You can override and "
            f"continue with the current corpus, but results will be mixed-model."
        )
    return None
