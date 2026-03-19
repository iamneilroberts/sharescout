# ShareScout — Knowledge Base Crawler

Network share / local filesystem document discovery tool. Crawls directories, scores files by relevance, extracts text, and uses local LLMs to summarize/categorize documents into a searchable catalog.

## Architecture

Modular Python package — `share_scout/` with separate modules per pipeline stage:

```
share_scout/
├── __main__.py      # CLI: crawl / web subcommands
├── config.py        # YAML config + CLI override loading
├── scanner.py       # Streaming os.walk with dir exclusions
├── scorer.py        # Additive scoring from scoring_rules.yaml
├── extractor.py     # Text extraction (txt/pdf/docx/xlsx) + SHA-256 partial hash
├── llm_client.py    # LLM client (Ollama or OpenAI-compatible) for summarization/categorization
├── catalog.py       # SQLite + FTS5 catalog (schema, CRUD, queries)
├── pipeline.py      # Orchestrate: scan → score → extract → LLM → catalog
├── checkpoint.py    # Resume support across crawl runs
└── web/
    ├── app.py       # Flask routes (dashboard, browse, search, file detail)
    ├── templates/   # Jinja2 (base, dashboard, browse, search, file_detail)
    └── static/      # style.css
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

# Web UI
python -m share_scout web
```

## Configuration

- `config.yaml` — root path, batch size, skip dirs, LLM endpoint/model, DB path, web host/port
- `scoring_rules.yaml` — extension scores, path pattern rules, size rules, score threshold
- `config.example.yaml` / `scoring_rules.example.yaml` — starter templates with documentation

## Data

- SQLite catalog: `share_scout.db` (in project root)
- Tables: `files`, `analyses`, `crawl_runs`, `analyses_fts` (FTS5)
- Checkpoint/resume: re-crawls skip already-processed files automatically

## Dependencies

Python 3.12+, venv at `.venv/`. Key packages: flask, pyyaml, ollama, python-docx, pymupdf, openpyxl, chardet.
