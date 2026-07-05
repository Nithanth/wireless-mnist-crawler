
from pathlib import Path
from typing import Optional

import typer


def register(app: typer.Typer, advanced: typer.Typer | None = None) -> None:
    _adv = advanced if advanced is not None else app

    @app.command("cache")
    def cache_cmd(
        action: str = typer.Argument("status", help="Action: status | inspect | gc | clear | clear-section | purge-venue"),
        section: Optional[str] = typer.Argument(None, help="Section name for clear-section (abstracts, dois, llm, oa, dataset_usage)"),
        cache_path: str = typer.Option(".wt_cache.json", "--cache-path"),
        keep_model: Optional[str] = typer.Option(
            None, "--keep-model",
            help="For gc: keep LLM entries whose recorded model matches this substring (e.g. 'gemini-3.5-flash'); everything else is pruned.",
        ),
        drop_unrecorded: bool = typer.Option(
            False, "--drop-unrecorded",
            help="For gc: also prune legacy LLM entries that never recorded a model.",
        ),
        venue: Optional[str] = typer.Option(None, "--venue", help="For purge-venue: venue name (e.g. SIGCOMM)."),
        year: Optional[int] = typer.Option(None, "--year", help="For purge-venue: year."),
        db: str = typer.Option("taxonomy.sqlite", "--db", help="For purge-venue: DB to look up the venue's papers."),
    ) -> None:
        """Inspect or manage the .wt_cache.json LLM/API response cache.

        \b
        Actions:
          status         Show entry counts per section and file size
          inspect        Show LLM entries grouped by the model that produced them
          gc             Prune LLM entries from other models (requires --keep-model)
          clear          Wipe the entire cache (prompts for confirmation)
          clear-section  Clear one section: abstracts | dois | llm | oa | dataset_usage
          purge-venue    Clear LLM extraction entries for one venue/year only
                         (requires --venue and --year; next run re-extracts
                         just those papers, everything else stays cached)
        """
        from wireless_taxonomy.analyze.cache import MetadataCache

        p = Path(cache_path)
        if not p.exists():
            typer.echo(f"Cache file not found: {p}")
            raise typer.Exit()

        c = MetadataCache(p)

        if action == "status":
            stats = c.stats()
            size_kb = p.stat().st_size / 1024
            typer.echo(f"Cache: {p}  ({size_kb:.1f} KB)")
            for section_name, count in stats.items():
                typer.echo(f"  {section_name:<20} {count} entries")

        elif action == "inspect":
            census = c.llm_model_census()
            if not census:
                typer.echo("LLM section is empty.")
                raise typer.Exit()
            typer.echo(f"LLM cache entries by model ({sum(census.values())} total):")
            for model, count in sorted(census.items(), key=lambda kv: -kv[1]):
                typer.echo(f"  {count:>6}×  {model}")

        elif action == "gc":
            if not keep_model:
                typer.echo("gc requires --keep-model <substring>, e.g. --keep-model gemini-3.5-flash", err=True)
                raise typer.Exit(1)
            census = c.llm_model_census()
            doomed_total = sum(
                n for m, n in census.items()
                if (m == "unrecorded" and drop_unrecorded)
                or (m != "unrecorded" and keep_model.strip().lower() not in m.lower())
            )
            if doomed_total == 0:
                typer.echo("Nothing to prune — all LLM entries already match.")
                raise typer.Exit()
            typer.confirm(
                f"Prune {doomed_total} LLM entries not matching '{keep_model}' from {p}?", abort=True
            )
            removed = c.gc_llm(keep_model, drop_unrecorded=drop_unrecorded)
            c.save()
            typer.echo(f"Pruned {removed} LLM entries. Kept models matching '{keep_model}'.")

        elif action == "clear":
            typer.confirm(f"Wipe ALL entries in {p}?", abort=True)
            c.clear()
            c.save()
            typer.echo("Cache cleared.")

        elif action == "clear-section":
            if not section:
                typer.echo("Provide a section name: abstracts | dois | llm | oa | dataset_usage", err=True)
                raise typer.Exit(1)
            try:
                removed = c.clear_section(section)
                c.save()
                typer.echo(f"Cleared {removed} entries from '{section}'.")
            except ValueError as exc:
                typer.echo(str(exc), err=True)
                raise typer.Exit(1)

        elif action == "purge-venue":
            if not venue or not year:
                typer.echo("purge-venue requires --venue and --year.", err=True)
                raise typer.Exit(1)
            import hashlib
            import sqlite3

            from wireless_taxonomy.analyze.dataset_extractor import _extraction_cache_key
            from wireless_taxonomy.config import load_settings
            from wireless_taxonomy.llm import LlmRouter

            conn = sqlite3.connect(db)
            conn.row_factory = sqlite3.Row
            papers = conn.execute(
                """
                SELECT p.id, p.title, p.abstract FROM papers p
                JOIN conference_instances ci ON ci.id = p.conference_instance_id
                JOIN venues v ON v.id = ci.venue_id
                WHERE LOWER(v.name) = LOWER(?) AND ci.year = ?
                """,
                (venue, year),
            ).fetchall()
            if not papers:
                typer.echo(f"No papers found for {venue} {year} in {db}.")
                raise typer.Exit()

            try:
                provider = LlmRouter(load_settings(db).llm).select_provider()
                model_identity = f"{provider.provider}/{provider.model}"
            except Exception:
                model_identity = ""

            doomed_keys: list[str] = []
            for paper in papers:
                artifacts = conn.execute(
                    "SELECT content_sha256 FROM paper_text_artifacts WHERE paper_id = ?",
                    (paper["id"],),
                ).fetchall()
                hashes = {a["content_sha256"][:16] for a in artifacts if a["content_sha256"]}
                for text in (paper["abstract"], paper["title"]):
                    if text:
                        hashes.add(hashlib.sha256(text.encode()).hexdigest()[:16])
                for h in hashes:
                    for mid in (model_identity, ""):
                        key = _extraction_cache_key(paper["id"], h, mid)
                        if c.get_llm(key) is not None:
                            doomed_keys.append(key)
            conn.close()

            removed = len(doomed_keys)
            if removed == 0:
                typer.echo(f"No cached extraction entries found for {venue} {year}.")
                raise typer.Exit()
            typer.confirm(
                f"Remove {removed} cached extraction entries for {venue} {year}? "
                f"(next run re-extracts those papers)", abort=True,
            )
            for key in doomed_keys:
                c.delete_llm(key)
            c.save()
            typer.echo(f"Purged {removed} extraction cache entries for {venue} {year}.")
            typer.echo(f"Re-run: extract-datasets --venue {venue} --years {year}")

        else:
            typer.echo(f"Unknown action '{action}'. Use: status | inspect | gc | clear | clear-section | purge-venue", err=True)
            raise typer.Exit(1)

    @_adv.command("cache-set-pdf")
    def cache_set_pdf(
        title: str = typer.Option(..., "--title", help="Exact paper title (used as the cache key)."),
        pdf_url: str = typer.Option(..., "--pdf-url", help="Direct URL to a legally hosted PDF of this paper."),
        doi: Optional[str] = typer.Option(None, "--doi", help="Paper DOI (also keyed in the cache if given)."),
        cache_path: str = typer.Option(".wt_cache.json", "--cache-path"),
        verify: bool = typer.Option(
            True, "--verify/--no-verify", help="Download the PDF and title-check it before trusting the URL."
        ),
    ) -> None:
        """Manually override the OA cache with a known-good PDF URL for one paper.

        For papers the OA indexes and web search miss (e.g. an author-hosted
        copy you found yourself). The URL is downloaded and title-verified
        first (unless --no-verify), then written to the OA cache so the next
        extract-datasets run fetches it like any other open-access paper.
        """
        from wireless_taxonomy.analyze.cache import MetadataCache

        if verify:
            from wireless_taxonomy.analyze.dataset_extractor import _fetch_pdf_bytes

            typer.echo(f"Verifying {pdf_url} ...")
            if _fetch_pdf_bytes(pdf_url, expected_title=title) is None:
                typer.echo(
                    "Verification FAILED: could not download a valid PDF whose first pages "
                    "contain this title. Check the URL (or use --no-verify to force).",
                    err=True,
                )
                raise typer.Exit(1)
            typer.echo("Verified: PDF downloads and title matches.")

        c = MetadataCache(cache_path)
        c.set_oa(
            title,
            doi,
            {
                "fetchable": True,
                "oa_status": "green",
                "license": "",
                "pdf_url": pdf_url,
                "provider": "manual",
                "source_url": pdf_url,
                "web_search_attempted": True,
            },
        )
        c.save()
        typer.echo(f"OA cache updated: '{title[:60]}' -> {pdf_url}")
