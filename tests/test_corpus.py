"""Tests for corpus management: auto-naming, snapshots, rollback, model guard."""

from pathlib import Path

from wireless_taxonomy.corpus import (
    Corpus,
    active_corpus,
    check_model_compatibility,
    list_corpora,
    next_auto_name,
    resolve_corpus,
    set_active,
)


class TestNameValidation:
    def test_path_traversal_rejected(self, tmp_path: Path) -> None:
        for bad in ("../evil", "a/b", "a\\b", ".hidden", " spaced", "sp ace"):
            try:
                Corpus(bad, tmp_path)
                raise AssertionError(f"expected ValueError for {bad!r}")
            except ValueError:
                pass

    def test_valid_names_accepted(self, tmp_path: Path) -> None:
        for good in ("corpus_v1", "mobicom-2025", "a.b.c", "V2"):
            Corpus(good, tmp_path)  # no raise


class TestAutoNaming:
    def test_first_corpus_is_v1(self, tmp_path: Path) -> None:
        c = resolve_corpus(None, create=True, root=tmp_path)
        assert c.name == "corpus_v1"

    def test_auto_name_increments(self, tmp_path: Path) -> None:
        resolve_corpus(None, create=True, root=tmp_path)
        assert next_auto_name(tmp_path) == "corpus_v2"

    def test_explicit_name(self, tmp_path: Path) -> None:
        c = resolve_corpus("mobicom_2025", create=True, root=tmp_path)
        assert c.name == "mobicom_2025"
        assert c.exists()

    def test_explicit_name_does_not_affect_auto_sequence(self, tmp_path: Path) -> None:
        resolve_corpus("custom", create=True, root=tmp_path)
        assert next_auto_name(tmp_path) == "corpus_v1"


class TestActiveCorpus:
    def test_new_corpus_becomes_active(self, tmp_path: Path) -> None:
        c = resolve_corpus(None, create=True, root=tmp_path)
        assert active_corpus(tmp_path).name == c.name

    def test_resolve_reuses_active(self, tmp_path: Path) -> None:
        c1 = resolve_corpus(None, create=True, root=tmp_path)
        c2 = resolve_corpus(None, create=True, root=tmp_path)
        assert c1.name == c2.name

    def test_set_active_switches(self, tmp_path: Path) -> None:
        resolve_corpus("a", create=True, root=tmp_path)
        resolve_corpus("b", create=True, root=tmp_path)
        set_active("a", tmp_path)
        assert active_corpus(tmp_path).name == "a"

    def test_no_active_returns_none(self, tmp_path: Path) -> None:
        assert active_corpus(tmp_path) is None


class TestModelGuard:
    def test_same_model_no_warning(self, tmp_path: Path) -> None:
        c = resolve_corpus(None, create=True, root=tmp_path)
        c.record_run("google/gemini-2.5-flash", "SIGCOMM 2024")
        assert check_model_compatibility(c, "google/gemini-2.5-flash") is None

    def test_different_model_warns(self, tmp_path: Path) -> None:
        c = resolve_corpus(None, create=True, root=tmp_path)
        c.record_run("google/gemini-2.5-flash", "SIGCOMM 2024")
        warning = check_model_compatibility(c, "anthropic/claude-sonnet-4")
        assert warning is not None
        assert "Mixing models" in warning

    def test_unrecorded_model_no_warning(self, tmp_path: Path) -> None:
        c = resolve_corpus(None, create=True, root=tmp_path)
        assert check_model_compatibility(c, "google/gemini-2.5-flash") is None

    def test_run_history_recorded(self, tmp_path: Path) -> None:
        c = resolve_corpus(None, create=True, root=tmp_path)
        c.record_run("google/gemini-2.5-flash", "SIGCOMM 2024")
        c.record_run("google/gemini-2.5-flash", "IMC 2024")
        runs = c.read_meta()["runs"]
        assert len(runs) == 2
        assert runs[1]["venues_years"] == "IMC 2024"


class TestSnapshots:
    def test_snapshot_none_without_db(self, tmp_path: Path) -> None:
        c = resolve_corpus(None, create=True, root=tmp_path)
        assert c.snapshot() is None

    def test_snapshot_and_rollback(self, tmp_path: Path) -> None:
        c = resolve_corpus(None, create=True, root=tmp_path)
        c.db_path.write_text("state A")
        snap = c.snapshot()
        c.db_path.write_text("state B")
        c.rollback(snap.stem)
        assert c.db_path.read_text() == "state A"

    def test_rollback_is_reversible(self, tmp_path: Path) -> None:
        c = resolve_corpus(None, create=True, root=tmp_path)
        c.db_path.write_text("state A")
        snap = c.snapshot()
        c.db_path.write_text("state B")
        c.rollback(snap.stem)
        # The pre-rollback state (B) was itself snapshotted
        snaps = c.list_snapshots()
        assert len(snaps) == 2

    def test_rollback_missing_snapshot_raises(self, tmp_path: Path) -> None:
        c = resolve_corpus(None, create=True, root=tmp_path)
        c.db_path.write_text("x")
        try:
            c.rollback("29990101")
            raise AssertionError("expected FileNotFoundError")
        except FileNotFoundError:
            pass

    def test_snapshot_collision_gets_suffix(self, tmp_path: Path) -> None:
        c = resolve_corpus(None, create=True, root=tmp_path)
        c.db_path.write_text("x")
        s1 = c.snapshot()
        s2 = c.snapshot()  # same second — must not overwrite
        assert s1 != s2

    def test_max_snapshots_pruned(self, tmp_path: Path) -> None:
        from wireless_taxonomy.corpus import MAX_SNAPSHOTS

        c = resolve_corpus(None, create=True, root=tmp_path)
        c.db_path.write_text("x")
        for _ in range(MAX_SNAPSHOTS + 3):
            c.snapshot()
        assert len(c.list_snapshots()) == MAX_SNAPSHOTS


class TestAdoptLegacy:
    def test_adopt_moves_db_and_results(self, tmp_path: Path) -> None:
        from wireless_taxonomy.corpus import adopt_legacy

        legacy_db = tmp_path / "taxonomy.sqlite"
        legacy_db.write_text("legacy db")
        legacy_results = tmp_path / "results"
        legacy_results.mkdir()
        (legacy_results / "a_datasets.csv").write_text("data")

        root = tmp_path / "corpora"
        c = adopt_legacy("wireless_v1", db_path=legacy_db,
                         results_path=legacy_results,
                         model_identity="google/gemini-2.5-flash", root=root)

        assert c.db_path.read_text() == "legacy db"
        assert (c.results_dir / "a_datasets.csv").read_text() == "data"
        assert not legacy_db.exists()  # moved, not copied
        assert c.model_identity() == "google/gemini-2.5-flash"
        assert active_corpus(root).name == "wireless_v1"

    def test_adopt_existing_corpus_fails(self, tmp_path: Path) -> None:
        from wireless_taxonomy.corpus import adopt_legacy

        root = tmp_path / "corpora"
        resolve_corpus("wireless_v1", create=True, root=root)
        legacy_db = tmp_path / "taxonomy.sqlite"
        legacy_db.write_text("x")
        try:
            adopt_legacy("wireless_v1", db_path=legacy_db, root=root)
            raise AssertionError("expected FileExistsError")
        except FileExistsError:
            pass

    def test_adopt_no_legacy_db_fails(self, tmp_path: Path) -> None:
        from wireless_taxonomy.corpus import adopt_legacy

        try:
            adopt_legacy("x", db_path=tmp_path / "missing.sqlite",
                         root=tmp_path / "corpora")
            raise AssertionError("expected FileNotFoundError")
        except FileNotFoundError:
            pass


class TestListCorpora:
    def test_empty(self, tmp_path: Path) -> None:
        assert list_corpora(tmp_path) == []

    def test_lists_created(self, tmp_path: Path) -> None:
        resolve_corpus("a", create=True, root=tmp_path)
        resolve_corpus("b", create=True, root=tmp_path)
        names = [c.name for c in list_corpora(tmp_path)]
        assert names == ["a", "b"]
