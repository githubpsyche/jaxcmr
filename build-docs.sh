#!/usr/bin/env bash
# Prepare generated chapters before Quarto checks the book's chapter list.
set -euo pipefail
cd "$(dirname "$0")"

uv run --no-sync python guide/prepare.py
quarto render guide
