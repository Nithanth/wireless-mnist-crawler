import pytest

from wireless_taxonomy.ingest.dblp import DblpIngestAdapter, stream_for_venue


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
