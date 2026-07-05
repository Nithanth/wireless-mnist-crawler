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


def parse_model_override(spec: str):
    """Parse a ``provider/model`` string into an LlmSettings override.

    Only the specified provider is configured; others are left unconfigured
    so the router uses exactly the requested model.  Raises BadParameter
    if the format is wrong.

    Example: ``google/gemini-2.0-flash`` → LlmSettings with google primary.
    """
    import os

    from wireless_taxonomy.config import LlmSettings, ProviderConfig

    parts = spec.strip().split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise typer.BadParameter(
            f"Model override must be provider/model (e.g. google/gemini-2.0-flash), got {spec!r}."
        )
    provider_name, model_name = parts[0].strip().lower(), parts[1].strip()
    valid = {"openai", "anthropic", "google"}
    if provider_name not in valid:
        raise typer.BadParameter(f"Provider must be one of {', '.join(sorted(valid))}, got {provider_name!r}.")

    key_env_map = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "google": "GEMINI_API_KEY"}
    key_env = key_env_map[provider_name]
    # For google, also check GOOGLE_API_KEY
    if provider_name == "google":
        api_key_configured = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    else:
        api_key_configured = bool(os.getenv(key_env))

    override = ProviderConfig(
        provider=provider_name,  # type: ignore[arg-type]
        model=model_name,
        api_key_env=key_env,
        api_key_configured=api_key_configured,
    )
    # Build a full LlmSettings so the router works — only the target provider
    # is configured; the rest are stubs so the router doesn't pick them.
    providers = {}
    for p in valid:
        if p == provider_name:
            providers[p] = override
        else:
            providers[p] = ProviderConfig(
                provider=p,  # type: ignore[arg-type]
                model="",
                api_key_env=key_env_map[p],
                api_key_configured=False,
            )
    return LlmSettings(
        primary_provider=provider_name,  # type: ignore[arg-type]
        fallback_providers=(),
        providers=providers,
    )


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
