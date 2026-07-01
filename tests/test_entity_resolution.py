"""Tests for entity resolution and dataset consolidation."""

import pytest

from wireless_taxonomy.postprocess.entity_resolution import (
    CanonicalDataset,
    DatasetRecord,
    LLMConfirmer,
    Match,
    SimilarityFlagger,
    URLDedup,
    consolidate,
    reconcile,
)


def _ds(name, keys=None, url="", modalities="", osi="", env=""):
    return DatasetRecord(
        name=name,
        bibtex_keys=keys or [],
        modalities=modalities,
        osi_layers=osi,
        environment=env,
        availability_url=url,
    )


class TestURLDedup:
    def test_shared_url(self):
        a = _ds("Dataset A", keys=["paper1"], url="https://github.com/org/repo")
        b = _ds("Dataset B", keys=["paper2"], url="https://github.com/org/repo/")
        matches = URLDedup().resolve([a, b])
        assert len(matches) == 1
        assert matches[0].confidence == 0.95
        assert matches[0].method == "url_dedup"

    def test_same_paper_not_matched(self):
        a = _ds("Dataset A", keys=["paper1"], url="https://example.com/data")
        b = _ds("Dataset B", keys=["paper1"], url="https://example.com/data")
        matches = URLDedup().resolve([a, b])
        assert len(matches) == 0

    def test_no_shared_url(self):
        a = _ds("Dataset A", keys=["p1"], url="https://example.com/a")
        b = _ds("Dataset B", keys=["p2"], url="https://example.com/b")
        matches = URLDedup().resolve([a, b])
        assert len(matches) == 0


class TestSimilarityFlagger:
    def test_high_name_similarity(self):
        a = _ds("5G Network Throughput Dataset", keys=["p1"], modalities="throughput", osi="L3")
        b = _ds("5G Network Throughput Traces", keys=["p2"], modalities="throughput", osi="L3")
        matches = SimilarityFlagger(name_threshold=0.75).resolve([a, b])
        assert len(matches) == 1

    def test_low_similarity_not_flagged(self):
        a = _ds("LoRa Indoor Propagation", keys=["p1"], modalities="RSSI", osi="L1")
        b = _ds("5G Satellite Orbit Data", keys=["p2"], modalities="TLE", osi="L3")
        matches = SimilarityFlagger(name_threshold=0.75).resolve([a, b])
        assert len(matches) == 0


class TestLLMConfirmer:
    def test_yes_verdict(self):
        def mock_complete(prompt):
            return '{"verdict": "yes", "reason": "Same dataset"}'

        confirmer = LLMConfirmer(llm_complete=mock_complete)
        candidate = Match(
            a=_ds("CelesTrak TLE", keys=["p1"]),
            b=_ds("Satellite Orbit Traces", keys=["p2"]),
            confidence=0.65,
            reason="name=0.60",
            method="similarity",
        )
        result = confirmer.confirm_pairs([candidate])
        assert len(result) == 1
        assert result[0].method == "llm_confirmed"
        assert result[0].confidence == 0.90

    def test_no_verdict_dropped(self):
        def mock_complete(prompt):
            return '{"verdict": "no", "reason": "Different measurement campaigns"}'

        confirmer = LLMConfirmer(llm_complete=mock_complete)
        candidate = Match(
            a=_ds("Starlink A", keys=["p1"]),
            b=_ds("Starlink B", keys=["p2"]),
            confidence=0.65,
            reason="name=0.70",
            method="similarity",
        )
        result = confirmer.confirm_pairs([candidate])
        assert len(result) == 0

    def test_unsure_verdict(self):
        def mock_complete(prompt):
            return '{"verdict": "unsure", "reason": "Possibly same"}'

        confirmer = LLMConfirmer(llm_complete=mock_complete)
        candidate = Match(
            a=_ds("WiFi Data", keys=["p1"]),
            b=_ds("Wi-Fi Dataset", keys=["p2"]),
            confidence=0.65,
            reason="name=0.70",
            method="similarity",
        )
        result = confirmer.confirm_pairs([candidate])
        assert len(result) == 1
        assert result[0].method == "llm_unsure"
        assert result[0].confidence == 0.70

    def test_malformed_response_becomes_unsure(self):
        def mock_complete(prompt):
            return "I think these are the same"  # no JSON

        confirmer = LLMConfirmer(llm_complete=mock_complete)
        candidate = Match(
            a=_ds("A", keys=["p1"]),
            b=_ds("B", keys=["p2"]),
            confidence=0.65,
            reason="test",
            method="similarity",
        )
        result = confirmer.confirm_pairs([candidate])
        assert len(result) == 1
        assert result[0].method == "llm_unsure"


class TestConsolidate:
    def test_no_matches_preserves_all(self):
        datasets = [
            _ds("Dataset A", keys=["p1"]),
            _ds("Dataset B", keys=["p2"]),
        ]
        result = consolidate(datasets, [])
        assert len(result) == 2

    def test_same_name_different_papers_merged(self):
        datasets = [
            _ds("WiFi-CSI Dataset", keys=["p1"], modalities="CSI"),
            _ds("WiFi-CSI Dataset", keys=["p2"], modalities="CSI; RSSI"),
        ]
        result = consolidate(datasets, [])
        assert len(result) == 1
        assert result[0].reuse_count == 2
        assert "p1" in result[0].bibtex_keys
        assert "p2" in result[0].bibtex_keys

    def test_high_confidence_match_merged(self):
        a = _ds("CelesTrak TLE Dataset", keys=["p1"], url="https://celestrak.org")
        b = _ds("Operational Satellite Orbits", keys=["p2"], url="https://celestrak.org")
        datasets = [a, b]
        matches = [Match(a=a, b=b, confidence=0.95, reason="shared URL", method="url_dedup")]
        result = consolidate(datasets, matches)
        assert len(result) == 1
        assert result[0].reuse_count == 2
        assert result[0].canonical_name == "Operational Satellite Orbits"  # longer name

    def test_low_confidence_not_merged(self):
        a = _ds("Dataset A", keys=["p1"])
        b = _ds("Dataset B", keys=["p2"])
        datasets = [a, b]
        matches = [Match(a=a, b=b, confidence=0.70, reason="unsure", method="llm_unsure")]
        result = consolidate(datasets, matches)
        assert len(result) == 2  # not merged (0.70 < 0.85 threshold)

    def test_reuse_count_sorting(self):
        datasets = [
            _ds("Popular Dataset", keys=["p1"]),
            _ds("Popular Dataset", keys=["p2"]),
            _ds("Popular Dataset", keys=["p3"]),
            _ds("Rare Dataset", keys=["p4"]),
        ]
        result = consolidate(datasets, [])
        assert result[0].canonical_name == "Popular Dataset"
        assert result[0].reuse_count == 3
        assert result[1].canonical_name == "Rare Dataset"
        assert result[1].reuse_count == 1


class TestReconcile:
    def test_url_dedup_takes_priority(self):
        a = _ds("A", keys=["p1"], url="https://example.com/data")
        b = _ds("B", keys=["p2"], url="https://example.com/data")
        matches = reconcile([a, b], url_dedup=True, similarity=False)
        assert len(matches) == 1
        assert matches[0].method == "url_dedup"

    def test_no_duplicate_pairs(self):
        a = _ds("5G Traces", keys=["p1"], url="https://example.com/5g", modalities="throughput", osi="L3")
        b = _ds("5G Trace Data", keys=["p2"], url="https://example.com/5g", modalities="throughput", osi="L3")
        matches = reconcile([a, b], url_dedup=True, similarity=True)
        # Should only appear once (URL dedup wins, similarity doesn't re-add)
        assert len(matches) == 1
        assert matches[0].method == "url_dedup"
