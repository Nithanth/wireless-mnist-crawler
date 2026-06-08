import pytest

from wireless_taxonomy.ingest.dblp import DblpIngestAdapter, is_non_paper_title, stream_for_venue


def test_is_non_paper_title_flags_posters_demos_workshops() -> None:
    assert is_non_paper_title("Poster: Measuring Internet Resilience")
    assert is_non_paper_title("Demo: A Real-Time SDR Platform")
    assert is_non_paper_title("Demonstration: Live 5G Testbed")
    assert is_non_paper_title("Work-in-Progress: Early Results")
    assert is_non_paper_title("Keynote: The Future of Networking")
    assert is_non_paper_title("Extended Abstract: Toward X")
    # Real papers (including ones whose topic mentions posters/demos) are kept.
    assert not is_non_paper_title("A Wireless CSI Sensing System")
    assert not is_non_paper_title("Demystifying the Mobile Control Plane")
    assert not is_non_paper_title("Posterior Sampling for Network Inference")


def test_dblp_adapter_drops_poster_and_demo_entries() -> None:
    page = {
        "result": {
            "hits": {
                "@total": "3",
                "hit": [
                    {"info": {"title": "A Real Main-Track Paper.", "year": "2024"}},
                    {"info": {"title": "Poster: A Short Poster.", "year": "2024"}},
                    {"info": {"title": "Demo: A Demonstration.", "year": "2024"}},
                ],
            }
        }
    }
    adapter = DblpIngestAdapter("IMC", 2024, fetch_json=lambda url: page, sleep_seconds=0)
    titles = [s.title for s in adapter.fetch()]
    assert titles == ["A Real Main-Track Paper"]


def test_stream_for_venue_known_and_unknown() -> None:
    assert stream_for_venue("SIGCOMM") == "conf/sigcomm/sigcomm"
    assert stream_for_venue("nsdi") == "conf/nsdi/nsdi"
    with pytest.raises(ValueError):
        stream_for_venue("Some Unknown Venue")


def test_dblp_adapter_parses_hits_and_paginates() -> None:
    page1 = {
        "result": {
            "hits": {
                "@total": "150",
                "hit": [
                    {
                        "info": {
                            "title": "A Wireless CSI Paper.",
                            "authors": {"author": [{"text": "Alice Smith 0001"}, {"text": "Bob Jones"}]},
                            "doi": "10.1145/ABC",
                            "year": "2024",
                            "type": "Conference and Workshop Papers",
                        }
                    }
                ],
            }
        }
    }
    page2 = {
        "result": {
            "hits": {
                "@total": "150",
                "hit": [
                    {
                        "info": {
                            "title": "Proceedings front matter",
                            "type": "Editorship",
                        }
                    },
                    {
                        "info": {
                            "title": "DOI From EE Paper",
                            "authors": {"author": {"text": "Carol Lin"}},
                            "ee": "https://doi.org/10.1109/XYZ.2024.123",
                            "year": "2024",
                        }
                    },
                ],
            }
        }
    }
    calls = []

    def fake_fetch(url: str) -> dict:
        calls.append(url)
        return page1 if "f=0" in url else page2

    adapter = DblpIngestAdapter("SIGCOMM", 2024, fetch_json=fake_fetch, sleep_seconds=0)
    seeds = adapter.fetch()

    assert len(calls) == 2  # paginated
    titles = [s.title for s in seeds]
    assert titles == ["A Wireless CSI Paper", "DOI From EE Paper"]  # editorship skipped, trailing dot stripped
    first = seeds[0]
    assert first.authors == ["Alice Smith", "Bob Jones"]  # disambiguation suffix stripped
    assert first.doi == "10.1145/abc"
    assert seeds[1].doi == "10.1109/xyz.2024.123"  # parsed from ee URL
