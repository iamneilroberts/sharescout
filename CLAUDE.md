# ShareScout — Knowledge Base Crawler

Network share / local filesystem document discovery tool. Crawls directories, scores files by relevance, extracts text, and uses local LLMs to summarize/categorize documents into a searchable catalog with RAG search and MCP server integration.

## Architecture

Modular Python package — `share_scout/` with separate modules per pipeline stage:

```
share_scout/
├── __main__.py      # CLI: crawl / web / ask subcommands
├── config.py        # YAML config + CLI override loading
├── scanner.py       # Streaming os.walk with dir exclusions
├── scorer.py        # Additive scoring from scoring_rules.yaml
├── extractor.py     # Text + image extraction (txt/pdf/docx/xlsx/pptx) + SHA-256 partial hash
├── analyzer.py      # Adaptive analysis strategy (single/sampled/chunked) + image captioning
├── llm_client.py    # LLM client (Ollama or OpenAI-compatible) for summarization/categorization/vision
├── embedder.py      # Vector embedding generation with chunking strategies
├── rag.py           # RAG search: embed query → vector KNN → LLM answer with sources
├── catalog.py       # SQLite + FTS5 + sqlite-vec catalog (schema, CRUD, queries, vector search)
├── pipeline.py      # Orchestrate: scan → score → extract → LLM → embed → catalog
├── checkpoint.py    # Resume support across crawl runs
├── prompts.py       # Prompt templates and preset management
├── mcp_server.py    # MCP server exposing search/ask/get_file/list_categories tools
└── web/
    ├── app.py       # Flask routes (dashboard, browse, search, file detail, etc.)
    ├── templates/   # Jinja2 templates
    └── static/      # CSS and JS
```

## Commands

```bash
# Activate venv
source .venv/bin/activate

# Dry run — score files, print distribution (no extraction or LLM)
python -m share_scout crawl --dry-run

# Full crawl (uses config.yaml for root_path, Ollama endpoint, etc.)
python -m share_scout crawl

# Full crawl with overrides
python -m share_scout crawl --root-path /some/path --ollama-model llama3.2

# RAG search
python -m share_scout ask "How does authentication work?"

# Web UI
python -m share_scout web
```

## Configuration

- `config.yaml` — root path, batch size, skip dirs, LLM endpoint/model, embedding model, DB path, web host/port
- `scoring_rules.yaml` — extension scores, path pattern rules, size rules, score threshold
- `project_groups.yaml` — organize projects into named groups
- `config.example.yaml` / `scoring_rules.example.yaml` / `project_groups.example.yaml` — starter templates

## Data

- SQLite catalog: `share_scout.db` (in project root)
- Tables: `files`, `analyses`, `crawl_runs`, `analyses_fts` (FTS5), `chunk_summaries`, `chunks_vec` (sqlite-vec), `embeddings_meta`
- Checkpoint/resume: re-crawls skip already-processed files automatically

## Dependencies

Python 3.12+, venv at `.venv/`. Key packages: flask, pyyaml, ollama, python-docx, pymupdf, openpyxl, chardet, sqlite-vec, mcp.
