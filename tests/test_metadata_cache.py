

def test_delete_llm(tmp_path):
    from wireless_taxonomy.analyze.cache import MetadataCache

    cache = MetadataCache(tmp_path / "c.json")
    cache.set_llm("k1", {"label": "yes"})
    cache.save()

    cache2 = MetadataCache(tmp_path / "c.json")
    assert cache2.delete_llm("k1") is True
    assert cache2.delete_llm("k1") is False  # already gone
    assert cache2.delete_llm("never-existed") is False
    assert cache2.dirty is True
    cache2.save()

    cache3 = MetadataCache(tmp_path / "c.json")
    assert cache3.get_llm("k1") is None


def test_corrupt_cache_file_warns_and_starts_empty(tmp_path, capsys):
    from wireless_taxonomy.analyze.cache import MetadataCache

    bad = tmp_path / "c.json"
    bad.write_text("{ not valid json !!!", encoding="utf-8")
    cache = MetadataCache(bad)
    assert cache.stats()["llm"] == 0
    captured = capsys.readouterr()
    assert "corrupt" in captured.err.lower()


def test_llm_model_census_and_gc(tmp_path):
    from wireless_taxonomy.analyze.cache import MetadataCache

    cache = MetadataCache(tmp_path / "c.json")
    cache.set_llm("k1", {"label": "yes", "model_version": "google:gemini-3.5-flash"})
    cache.set_llm("k2", {"label": "no", "model_version": "llm_candidate_v0:google:gemini-3.5-flash"})
    cache.set_llm("k3", {"label": "yes", "model_version": "openai:gpt-5.4-mini"})
    cache.set_llm("k4", {"label": "maybe"})  # legacy, no model recorded

    census = cache.llm_model_census()
    assert census["google:gemini-3.5-flash"] == 1
    assert census["openai:gpt-5.4-mini"] == 1
    assert census["unrecorded"] == 1

    # gc keeps substring matches + unrecorded by default
    removed = cache.gc_llm("gemini-3.5-flash")
    assert removed == 1  # only the openai entry
    assert cache.get_llm("k1") is not None
    assert cache.get_llm("k2") is not None
    assert cache.get_llm("k3") is None
    assert cache.get_llm("k4") is not None

    # drop_unrecorded also prunes legacy entries
    removed = cache.gc_llm("gemini-3.5-flash", drop_unrecorded=True)
    assert removed == 1
    assert cache.get_llm("k4") is None
