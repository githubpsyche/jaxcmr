#!/usr/bin/env bash
# Prepare generated pages before Quarto discovers the website's render targets.
set -euo pipefail
cd "$(dirname "$0")"

uv run --no-sync python guide/prepare.py
quarto render guide
