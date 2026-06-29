"""Shared helpers used across CLI command modules."""

import typer

from wireless_taxonomy.config import load_settings
from wireless_taxonomy.pipeline import Pipeline

# Fields written by `classify --csv`; consumed by `eval`.
CSV_FIELDS = ["title", "authors", "doi", "venue", "year", "label", "confidence", "used_abstract", "has_abstract"]


def make_pipeline(db: str) -> Pipeline:
    return Pipeline(load_settings(db))


def parse_years(years: str) -> list[int]:
    """Parse ``2024`` or an inclusive range ``2023:2025`` into a list of years."""
    text = years.strip()
    if ":" in text:
        start_s, _, end_s = text.partition(":")
        try:
            start, end = int(start_s), int(end_s)
        except ValueError as exc:
            raise typer.BadParameter("--years range must look like 2023:2025.") from exc
        if start > end:
            raise typer.BadParameter("--years range start must be <= end.")
        return list(range(start, end + 1))
    try:
        return [int(text)]
    except ValueError as exc:
        raise typer.BadParameter("--years must be a year (2024) or range (2023:2025).") from exc


def parse_venue_years(entries: list[str]) -> list[tuple[str, str]]:
    """Parse ``VENUE:YEAR`` entries into ``(venue, year)`` pairs."""
    parsed: list[tuple[str, str]] = []
    for raw in entries:
        venue, sep, year = raw.partition(":")
        if not sep or not venue.strip() or not year.strip():
            raise typer.BadParameter(f"--exclude must look like VENUE:YEAR (got {raw!r}).")
        parsed.append((venue.strip(), year.strip()))
    return parsed


def pct(count: int, total: int) -> float:
    return round(100.0 * count / total, 1) if total else 0.0


def echo_breakdown(result: dict) -> None:
    counts = result["counts"]
    total = result["total_papers"]
    with_abs = result["papers_with_abstract"]
    abs_pct = pct(with_abs, total)
    typer.echo(
        f"{result['venue']} {result['year']} — {total} papers "
        f"(abstracts: {with_abs}/{total}, {abs_pct:.0f}%)"
    )
    for label in ("yes", "maybe", "no"):
        n = counts.get(label, 0)
        typer.echo(f"  {label:<5} {n:>4}  ({pct(n, total):>5.1f}%)")
