#!/bin/bash
# Build script for jaxcmr documentation
#
# Usage:
#   ./scripts/build-docs.sh        # Full build
#   ./scripts/build-docs.sh quick  # Main site only (faster)
#
# For incremental development:
#   quarto preview                           # Live preview main site
#   cd projects/TalmiEEG && quarto render    # Single project

cd "$(dirname "$0")/.." || exit 1

echo "Building jaxcmr documentation..."
echo ""

# Render main site
echo "1. Rendering main site..."
if quarto render 2>&1 | grep -q "Output created"; then
    echo "   Main site: OK"
else
    echo "   Main site: FAILED"
    exit 1
fi

# Quick mode - skip projects
if [ "$1" = "quick" ]; then
    echo ""
    echo "Quick mode - skipping projects"
    echo "Done! docs/index.html"
    exit 0
fi

# Create projects directory
mkdir -p docs/projects

# Render project subsites
echo ""
echo "2. Rendering project subsites..."

for project in projects/TalmiEEG projects/repfr projects/cru_to_cmr; do
    if [ -f "$project/_quarto.yml" ]; then
        name=$(basename "$project")
        echo ""
        echo "   $name:"

        # HTML
        if (cd "$project" && quarto render --to html 2>&1 | grep -q "Output created"); then
            echo "     HTML: OK"
        else
            echo "     HTML: FAILED"
        fi

        # PDF
        if (cd "$project" && quarto render --to apaquarto-pdf 2>&1 | grep -q "Output created"); then
            echo "     PDF: OK"
        else
            echo "     PDF: skipped"
        fi

        # DOCX
        if (cd "$project" && quarto render --to apaquarto-docx 2>&1 | grep -q "Output created"); then
            echo "     DOCX: OK"
        else
            echo "     DOCX: skipped"
        fi
    fi
done

echo ""
echo "=========================================="
echo "Build complete!"
echo "Main site:  docs/index.html"
echo "Projects:   docs/projects/*/"
echo "=========================================="
