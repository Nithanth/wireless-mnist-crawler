from wireless_taxonomy.evaluate.matching import make_record, match_papers


def test_exact_normalized_title_matches() -> None:
    manual = [make_record("Deep Learning for Radio")]
    automated = [make_record("deep   learning  for radio")]  # whitespace/case differences only
    result = match_papers(manual, automated)
    assert [pair.method for pair in result.matched] == ["exact"]
    assert result.missed_by_cli == []
    assert result.extra_from_cli == []


def test_fuzzy_matches_near_duplicate_title_without_authors() -> None:
    manual = [make_record("Deep Learning for Radio Maps")]
    automated = [make_record("Deep Learning for Radio Map")]  # singular vs plural
    result = match_papers(manual, automated)
    assert len(result.matched) == 1
    pair = result.matched[0]
    assert pair.method == "fuzzy"
    assert pair.title_similarity >= 0.92

    # With fuzzy disabled the near-duplicate is not matched.
    strict = match_papers(manual, automated, fuzzy=False)
    assert strict.matched == []
    assert strict.missed_by_cli == ["Deep Learning for Radio Maps"]
    assert strict.extra_from_cli == ["Deep Learning for Radio Map"]


def test_author_overlap_boosts_sub_threshold_title() -> None:
    manual = [make_record("alpha beta gamma", "Ada Lovelace, Grace Hopper")]
    automated = [make_record("alpha beta delta", "A. Lovelace, G. Hopper")]
    # Title alone is well below the strict threshold; only the author boost can match it.
    boosted = match_papers(
        manual,
        automated,
        title_threshold=0.99,
        author_boost_title_threshold=0.5,
        author_boost_min_overlap=0.5,
    )
    assert len(boosted.matched) == 1
    pair = boosted.matched[0]
    assert pair.method == "fuzzy"
    assert pair.author_overlap == 1.0
    assert pair.shared_authors == ["hopper", "lovelace"]

    # Same titles without authors: no boost, so no match under the strict title threshold.
    no_authors = match_papers(
        [make_record("alpha beta gamma")],
        [make_record("alpha beta delta")],
        title_threshold=0.99,
        author_boost_title_threshold=0.5,
        author_boost_min_overlap=0.5,
    )
    assert no_authors.matched == []


def test_matching_is_one_to_one_greedy() -> None:
    # Two manual papers both resemble a single automated paper; only the best matches.
    manual = [make_record("network telemetry system"), make_record("network telemetry systems")]
    automated = [make_record("network telemetry systems")]
    result = match_papers(manual, automated)
    assert len(result.matched) == 1
    assert len(result.missed_by_cli) == 1
    assert result.extra_from_cli == []
