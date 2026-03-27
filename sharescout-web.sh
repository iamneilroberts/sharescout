#!/usr/bin/env bash
# Launch the ShareScout web admin UI
cd "$(dirname "$0")"
source .venv/bin/activate
exec python -m share_scout web "$@"
