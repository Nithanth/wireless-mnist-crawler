#!/usr/bin/env bash
set -euo pipefail

# DEPRECATED: Use `wt add` instead.
#
# This script is kept for backward compatibility only.
# The CLI now handles everything:
#
#   wt add --venues SIGCOMM,IMC,NSDI --years 2022:2025 --workers 6
#   wt export
#
# See `wt --help` for the full workflow.

echo "DEPRECATED: Use 'wt add' instead."
echo ""
echo "  wt add --venues SIGCOMM,IMC,NSDI --years 2022:2025 --workers 6"
echo "  wt export"
echo ""
echo "Forwarding to 'wt add'..."
echo ""

# Forward arguments to wt add
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.venv/bin/activate"
exec python -m wireless_taxonomy.cli add "$@"
