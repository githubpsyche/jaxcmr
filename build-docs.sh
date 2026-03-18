#!/bin/bash
# Build script for jaxcmr documentation
#
# Usage:
#   ./build-docs.sh        # Full build
#   ./build-docs.sh quick  # Main site only (faster)
#
# For incremental development:
#   quarto preview                           # Live preview main site

echo "Building jaxcmr documentation..."
echo ""

# Render main site
echo "1. Rendering main site..."
echo ""
if quarto render --no-clean; then
    echo ""
    echo "   Main site: OK"
else
    echo ""
    echo "   Main site: FAILED"
    exit 1
fi

echo ""
echo "=========================================="
echo "Build complete!"
echo "Main site:  docs/index.html"
echo "=========================================="
