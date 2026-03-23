# ShareScout MCP Server + Nightly Cron

## Summary

Expose the ShareScout knowledge base as a local MCP server (stdio transport) and add a nightly cron job to keep the catalog updated.

## MCP Server

**File:** `share_scout/mcp_server.py`
**Transport:** stdio
**Launch script:** `sharescout-mcp.sh`
**Dependency:** `mcp` Python package (FastMCP)

### Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `search` | Full-text search over file summaries/keywords via FTS5 | `query` (str), `limit` (int, default 10) |
| `ask` | RAG query with semantic search + LLM-generated answer | `question` (str), `mode` ("query"\|"chat", default "query") |
| `get_file` | Full metadata + analysis for a specific file | `file_id` (int) |
| `list_categories` | All categories with file counts | none |

### Implementation

- Each tool is a thin wrapper around existing code in `catalog.py` and `rag.py`
- Server loads config from `config.yaml` to get db_path and Ollama settings
- No new database tables or schema changes

### Launch Script (`sharescout-mcp.sh`)

```bash
#!/usr/bin/env bash
cd "$(dirname "$0")"
source .venv/bin/activate
exec python -m share_scout.mcp_server "$@"
```

### Claude Code Config (`~/.claude/settings.local.json`)

```json
{
  "mcpServers": {
    "sharescout": {
      "command": "/home/neil/dev/kbase/sharescout-mcp.sh"
    }
  }
}
```

## Nightly Update Script

**File:** `sharescout-update.sh`

- Activates venv, runs `python -m share_scout crawl`
- `--scan-only` flag skips LLM/embedding phases (fast filesystem pass only)
- Default: full pipeline (scan + LLM + embed)
- Checks if Ollama is reachable before LLM phase; falls back to scan-only with warning if not
- Logs to `~/.local/share/sharescout/nightly.log` (rotates, keeps last 7 days)

**Cron entry:** `0 2 * * * /home/neil/dev/kbase/sharescout-update.sh`

## Files to Create

| File | Description |
|------|-------------|
| `share_scout/mcp_server.py` | MCP server module |
| `sharescout-mcp.sh` | MCP launch script |
| `sharescout-update.sh` | Nightly crawl script |

## Files to Modify

None. All functionality reuses existing modules via imports.

## Not In Scope

- No HTTP/SSE transport
- No write operations from MCP (no triggering crawls)
- No new database schema
- No changes to existing modules
