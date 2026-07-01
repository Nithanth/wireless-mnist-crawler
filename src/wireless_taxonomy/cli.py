"""Wireless paper classification + dataset extraction CLI.

Entry point: `wireless-taxonomy` (via pyproject.toml) or
`python -m wireless_taxonomy.cli`.

Command implementations live in `wireless_taxonomy/commands/`.
This module owns the Typer app object, the Typer/Click compatibility
patch (needed for Click 8.2 + Typer 0.15.x), and wires all commands in.
"""

import inspect

import click
import typer
from typer.core import TyperArgument, TyperOption

# ── Typer / Click 8.2 compatibility patch ────────────────────────────────────
# Typer 0.15.x rich help calls make_metavar without Click 8.2's required `ctx`
# argument. Patch all affected classes once at import time.

_OPTION_MAKE_METAVAR = TyperOption.make_metavar
_ARGUMENT_MAKE_METAVAR = TyperArgument.make_metavar
_CLICK_PARAMETER_MAKE_METAVAR = click.core.Parameter.make_metavar
_CLICK_OPTION_MAKE_METAVAR = click.core.Option.make_metavar
_CLICK_ARGUMENT_MAKE_METAVAR = click.core.Argument.make_metavar


def _patch_typer_click_compat() -> None:
    for cls, original in [
        (click.core.Parameter, _CLICK_PARAMETER_MAKE_METAVAR),
        (click.core.Option, _CLICK_OPTION_MAKE_METAVAR),
        (click.core.Argument, _CLICK_ARGUMENT_MAKE_METAVAR),
    ]:
        params = inspect.signature(cls.make_metavar).parameters
        if params.get("ctx") is not None and params["ctx"].default is inspect.Parameter.empty:

            def make_metavar(self, ctx=None, _original=original):
                return _original(self, ctx)

            cls.make_metavar = make_metavar  # type: ignore[method-assign]

    option_params = inspect.signature(TyperOption.make_metavar).parameters
    if option_params.get("ctx") is not None and option_params["ctx"].default is inspect.Parameter.empty:

        def option_make_metavar(self, ctx=None):
            return _OPTION_MAKE_METAVAR(self, ctx)

        TyperOption.make_metavar = option_make_metavar  # type: ignore[method-assign]

    argument_params = inspect.signature(TyperArgument.make_metavar).parameters
    if argument_params.get("ctx") is None:

        def argument_make_metavar(self, ctx=None):
            if self.metavar is not None:
                return self.metavar
            var = (self.name or "").upper()
            if not self.required:
                var = f"[{var}]"
            type_var = self.type.get_metavar(param=self, ctx=ctx)
            if type_var:
                var += f":{type_var}"
            if self.nargs != 1:
                var += "..."
            return var

        TyperArgument.make_metavar = argument_make_metavar  # type: ignore[method-assign]


_patch_typer_click_compat()

# ── App ───────────────────────────────────────────────────────────────────────

app = typer.Typer(
    help=(
        "Wireless paper classification + dataset extraction CLI.\n\n"
        "Commands:\n"
        "  classify          Classify papers as wireless (yes/maybe/no) for a venue/year.\n"
        "  eval              DB-free snapshot eval of classified CSV vs gold sheet.\n"
        "  fetch-coverage    Report OA full-text availability per venue/year.\n"
        "  extract-datasets  Full pipeline: classify → fetch PDF → extract datasets → CSV.\n"
        "  merge-results     Combine per-venue/year CSVs into master files.\n"
        "  cache             Inspect or clear the LLM/API cache.\n"
        "  llm-config        Show configured LLM providers and models."
    )
)

# ── Register commands (import after app is defined to avoid circular deps) ────

from wireless_taxonomy.commands import admin, cache, classify, coverage, eval, extract, merge, reconcile  # noqa: E402
from wireless_taxonomy.commands._shared import parse_venue_years as _parse_venue_years  # noqa: F401 (re-exported for tests)

classify.register(app)
eval.register(app)
coverage.register(app)
extract.register(app)
merge.register(app)
cache.register(app)
admin.register(app)
reconcile.register(app)

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app()
