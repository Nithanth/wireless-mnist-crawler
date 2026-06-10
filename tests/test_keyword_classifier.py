from wireless_taxonomy.analyze.candidates import KeywordCandidateClassifier


def _classify(title: str, abstract: str | None = None) -> object:
    return KeywordCandidateClassifier().classify({"id": 1, "title": title, "abstract": abstract})


def test_wireless_title_is_yes_with_category_evidence() -> None:
    pred = _classify("Beamforming for 5G MIMO Systems")
    assert pred.label == "yes"
    assert pred.confidence >= 0.91
    assert "category=wireless" in pred.evidence
    assert "beamforming" in pred.evidence


def test_word_boundary_avoids_substring_false_positives() -> None:
    # "rf" must not match "surfing"; "ran" must not match "branch".
    pred = _classify("Surfing the branch predictor for faster compilers")
    assert pred.label != "yes"
    assert "category=wireless" not in pred.evidence


def test_networking_paper_is_no() -> None:
    pred = _classify("Datacenter congestion control for TCP traffic")
    assert pred.label == "no"
    assert "category=networking_non_wireless" in pred.evidence


def test_computing_paper_is_no() -> None:
    pred = _classify("A new operating system kernel for GPU databases")
    assert pred.label == "no"
    assert "category=not_relevant" in pred.evidence


def test_uncertain_when_no_terms_match() -> None:
    pred = _classify("A study of human behavior in markets")
    assert pred.label == "maybe"
    assert "category=uncertain" in pred.evidence


def test_normalization_matches_wifi_and_mmwave_variants() -> None:
    assert _classify("Wi-Fi sensing indoors").label == "yes"
    assert _classify("Millimeter wave links").label == "yes"


def test_confidence_scales_with_matched_terms() -> None:
    one = _classify("A wireless study")
    many = _classify("5G mmWave MIMO beamforming with OFDM")
    assert many.confidence > one.confidence
    assert many.confidence <= 0.98


def test_missing_abstract_flagged_in_evidence_and_lowers_confidence() -> None:
    with_abstract = _classify("Routing in networks", abstract="We study packet routing.")
    without_abstract = _classify("Routing in networks", abstract=None)
    assert "abstract_missing=true" in without_abstract.evidence
    assert "abstract_missing=true" not in with_abstract.evidence
    assert without_abstract.confidence < with_abstract.confidence


def test_used_abstract_flag() -> None:
    assert _classify("5G", abstract="real abstract").used_abstract is True
    assert _classify("5G", abstract=None).used_abstract is False
