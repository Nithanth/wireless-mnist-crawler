
from pathlib import Path
from typing import Optional

import typer


def register(app: typer.Typer) -> None:
    @app.command("cache")
    def cache_cmd(
        action: str = typer.Argument("status", help="Action: status | clear | clear-section"),
        section: Optional[str] = typer.Argument(None, help="Section name for clear-section (abstracts, dois, llm, oa, dataset_usage)"),
        cache_path: str = typer.Option(".wt_cache.json", "--cache-path"),
    ) -> None:
        """Inspect or manage the .wt_cache.json LLM/API response cache.

        \b
        Actions:
          status         Show entry counts per section and file size
          clear          Wipe the entire cache (prompts for confirmation)
          clear-section  Clear one section: abstracts | dois | llm | oa | dataset_usage
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

        else:
            typer.echo(f"Unknown action '{action}'. Use: status | clear | clear-section", err=True)
            raise typer.Exit(1)

    @app.command("cache-set-pdf")
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
