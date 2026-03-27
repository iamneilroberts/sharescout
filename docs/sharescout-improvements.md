# ShareScout: RAG, Vision, and MCP Server

ShareScout started as a filesystem crawler that scored and cataloged documents with LLM-generated summaries. Three additions turn it into something substantially more useful.

## What changed

### RAG / Semantic Search

Documents are now embedded with `nomic-embed-text` via Ollama and stored as vectors in SQLite using the `sqlite-vec` extension. Large documents are split into overlapping chunks (4000 chars, 400 overlap) with intelligent boundary detection. Very large files use a sampling strategy that prioritizes structural content — headings, code blocks, key-value pairs — over raw sequential reading.

Search works in two modes: **FTS5** for keyword queries against summaries/keywords, and **vector KNN** for semantic similarity. The `ask` command embeds your question, retrieves the nearest chunks, and feeds them to the LLM with source attribution. In `query` mode it returns extractive answers pinned to numbered source sentences; in `chat` mode it gives freeform answers grounded in retrieved context.

### Image Captioning

Documents containing images (PDFs, DOCX, PPTX) now have their images extracted, deduplicated by content hash, and captioned using a vision-capable model. Captions are inserted inline at the correct character offset so the LLM sees `[Image: architecture diagram showing three-tier deployment]` in context rather than losing visual information entirely. Small images (<100x100px) are skipped to avoid noise from logos and icons.

### MCP Server

An MCP server exposes four tools over stdio: `search`, `ask`, `get_file`, and `list_categories`. Claude Code loads it automatically from `.mcp.json` in the project root. This makes the entire catalog — summaries, keywords, categories, and RAG answers — available as tool calls within any Claude Code session working in this directory.

## How search improves inside Claude Code

Without ShareScout, Claude Code can grep files and read them one at a time. With the MCP tools, it can query a pre-built index of thousands of analyzed documents in seconds.

### Example use cases

**Finding implementations across projects:**
> "Which projects implement WebSocket connections?"

Claude calls `search("WebSocket implementation")` and gets back matching files with summaries and paths — no need to grep across 33,000+ files.

**Asking architectural questions:**
> "How does the auth flow work in the travel assistant?"

Claude calls `ask("How does authentication work in the travel assistant?")` and gets a sourced answer synthesized from the relevant files, with paths it can then read for detail.

**Discovering what exists:**
> "What database migration files do we have?"

Claude calls `search("database migration schema")` or `list_categories()` to see what's cataloged under "Database", then drills into specific files with `get_file()`.

**Cross-project comparison:**
> "Do any projects use the same API patterns as the Voygent backend?"

Claude calls `ask("API patterns similar to Voygent backend")` and gets semantically relevant matches even if they don't share exact keywords.

**Locating documentation:**
> "Find the design spec for the notification system"

Claude calls `search("notification system design spec")` — FTS5 matches against summaries that the LLM wrote during crawl, which are far more descriptive than filenames alone.

The key difference: ShareScout's catalog contains LLM-written summaries of every file, so searches match against *what files are about*, not just what text they contain.
