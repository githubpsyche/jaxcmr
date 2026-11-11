#!/usr/bin/env bash
# Stage matching notebook sources and inputs, then render their saved outputs.
set -euo pipefail
cd "$(dirname "$0")"

uv run --no-sync python prepare-guide.py stage "$@"
quarto render guide --no-execute
touch docs/.nojekyll
printf 'Documentation rendered to docs/index.html\n'
