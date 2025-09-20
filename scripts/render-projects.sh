#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

for d in projects/*/ ; do
  if [[ -f "${d}_quarto.yml" ]]; then
    echo "Rendering ${d}"
    quarto render "$d" --to html
  fi
done
