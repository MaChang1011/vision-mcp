#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate

if ! python -c "import paddleocr" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -e ".[ocr,opencv,detection,dev]" -q
fi

echo "Starting Vision MCP Server..."
exec python -m vision_mcp
