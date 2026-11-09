#!/usr/bin/env bash
# Stage the book's selected notebooks and inputs, then render saved outputs.
set -euo pipefail
cd "$(dirname "$0")"

# These directories contain generated copies only; authoritative notebooks stay
# in templates/. Keep chapter paths as plain '- templates/name.ipynb' entries.
rm -rf -- guide/templates guide/downloads _site/guide
mkdir -p guide/templates/data/fits guide/downloads
notebooks=()
while IFS= read -r notebook; do
    notebooks+=("$notebook")
    cp "${notebook}" "guide/${notebook}"
    cp "$notebook" "guide/downloads/${notebook##*/}"
done < <(sed -nE 's/^[[:space:]]*-[[:space:]]+(templates\/[A-Za-z0-9_]+\.ipynb)[[:space:]]*$/\1/p' guide/_quarto.yml)
if [ "${#notebooks[@]}" -eq 0 ]; then
    printf 'No notebook chapters found in guide/_quarto.yml\n' >&2
    exit 1
fi

# One observation source and its existing matrix, shared by all chapters.
for asset in Dupertuys2026.h5 Dupertuys2026_semantic_distances.npy \
    fits/Dupertuys2026_CMR_demo.json fits/Dupertuys2026_eCMR_source_only.json \
    fits/Dupertuys2026_eCMR_broad.json fits/README.md; do
    cp "data/$asset" "guide/templates/data/$asset"
done

quarto render guide --no-execute

printf 'Documentation rendered to _site/guide/index.html\n'
