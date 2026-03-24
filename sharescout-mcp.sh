#!/usr/bin/env bash
# Launch the ShareScout MCP server (stdio transport)
cd "$(dirname "$0")"
source .venv/bin/activate
exec python -m share_scout.mcp_server "$@"
