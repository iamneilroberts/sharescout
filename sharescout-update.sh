#!/usr/bin/env bash
# ShareScout nightly catalog update
# Usage: sharescout-update.sh [--scan-only]
#   --scan-only  Skip LLM analysis and embedding (fast filesystem scan only)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$HOME/.local/share/sharescout"
LOG_FILE="$LOG_DIR/nightly.log"

mkdir -p "$LOG_DIR"

# Rotate logs — keep last 7 days
find "$LOG_DIR" -name "nightly-*.log" -mtime +7 -delete 2>/dev/null || true
# Archive current log if it exists
if [ -f "$LOG_FILE" ]; then
    mv "$LOG_FILE" "$LOG_DIR/nightly-$(date +%Y%m%d-%H%M%S).log"
fi

cd "$SCRIPT_DIR"
source .venv/bin/activate

SCAN_ONLY=false
for arg in "$@"; do
    case "$arg" in
        --scan-only) SCAN_ONLY=true ;;
    esac
done

{
    echo "=== ShareScout nightly update: $(date) ==="

    if [ "$SCAN_ONLY" = true ]; then
        echo "Mode: scan-only (skipping LLM analysis and embedding)"
        python -m share_scout crawl --dry-run
        echo "Scan complete."
    else
        # Check if Ollama is reachable
        OLLAMA_ENDPOINT=$(python -c "
from share_scout.config import load_config
c = load_config('config.yaml')
print(c.get('ollama', {}).get('endpoint', 'http://localhost:11434'))
" 2>/dev/null || echo "http://localhost:11434")

        if curl -s --connect-timeout 5 "$OLLAMA_ENDPOINT/api/tags" > /dev/null 2>&1; then
            echo "Mode: full pipeline (Ollama reachable at $OLLAMA_ENDPOINT)"
            python -m share_scout crawl
            echo "Crawl complete. Starting embedding..."
            python -m share_scout embed
            echo "Embedding complete."
        else
            echo "WARNING: Ollama not reachable at $OLLAMA_ENDPOINT — falling back to scan-only"
            python -m share_scout crawl --dry-run
            echo "Scan-only complete (Ollama unavailable)."
        fi
    fi

    echo "=== Finished: $(date) ==="
} >> "$LOG_FILE" 2>&1
