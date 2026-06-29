
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
