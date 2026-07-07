"""Wireless paper classification + dataset extraction CLI.

Entry point: `wt` (via pyproject.toml). The legacy name `wireless-taxonomy`
still works as an alias. The module can also be run as
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
        "wt — wireless dataset extraction tool\n\n"
        "Workflow:\n"
        "  wt init                                           # create a corpus\n"
        "  wt add --venues SIGCOMM,NSDI --years 2022:2025   # extract data\n"
        "  wt export                                        # reconcile + final CSVs\n"
        "  wt status                                        # see what's in the corpus\n\n"
        "Use `wt advanced` for cache management, individual pipeline stages, and debugging."
    ),
    no_args_is_help=True,
)

# Advanced subgroup — individual stages, cache, DB tooling.
advanced = typer.Typer(
    help=(
        "Advanced: individual pipeline stages, cache management, DB tooling.\n\n"
        "These are the building blocks that `add` and `export` orchestrate.\n"
        "Use them for debugging, re-running a single stage, or fine-grained control."
    ),
    no_args_is_help=True,
)
app.add_typer(advanced, name="advanced")

# ── Register commands (import after app is defined to avoid circular deps) ────

from wireless_taxonomy.commands import admin, cache, classify, corpus_cmd, coverage, eval, eval_db, export, extract, merge, reconcile, report, run  # noqa: E402
from wireless_taxonomy.commands._shared import parse_venue_years as _parse_venue_years  # noqa: F401 (re-exported for tests)

# ── Primary commands (the 5 a user needs) ─────────────────────────────────────
run.register(app)        # `wt add`
export.register(app)     # `wt export`
corpus_cmd.register(app) # `wt init`, `wt status`, `wt rollback`

# ── Advanced commands (individual stages + tooling) ───────────────────────────
classify.register(advanced)
eval.register(advanced)
eval_db.register(advanced)
coverage.register(advanced)
extract.register(advanced)
merge.register(advanced)
reconcile.register(advanced)
report.register(advanced)
admin.register(advanced, advanced=advanced)
cache.register(advanced, advanced=advanced)

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app()
