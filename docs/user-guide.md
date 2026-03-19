# ShareScout User Guide

ShareScout crawls your project directories, scores files by relevance, extracts text, and uses a local Ollama LLM to summarize and categorize everything into a searchable catalog.

## Quick Start

```bash
cd sharescout
source .venv/bin/activate

# Start the web UI
python -m share_scout web

# Open http://localhost:8080
```

## Running a Crawl

### From the Web UI

Click the **Start Crawl** button on the Dashboard. The nav bar shows live progress while a crawl is running. Click **Stop Crawl** to interrupt — progress is saved and the next crawl resumes where it left off.

### From the Command Line

```bash
# Full crawl
python -m share_scout crawl

# Dry run — score files without extracting or calling the LLM
python -m share_scout crawl --dry-run

# Override settings
python -m share_scout crawl --root-path /some/other/path --ollama-model mistral:7b
```

### LLM Setup

ShareScout supports two LLM providers. See the [README](../README.md) for setup instructions.

- **Ollama** — Free, local, private. Install from ollama.com, pull a model, configure in config.yaml.
- **OpenAI-compatible API** — Faster, cloud-based. Set API key and endpoint in config.yaml.

If no LLM is reachable, the crawl still runs Phase 1 (scan + extract) and skips analysis. Re-run once your LLM is available.

### What Gets Crawled

The crawl runs in two phases:

- **Phase 1 (fast):** Walks the filesystem, scores each file, extracts text from qualifying files. Skips files already in the database.
- **Phase 2 (slow):** Sends extracted text to Ollama for summarization, keyword extraction, and categorization. Processes highest-scored files first.

Files are scored additively based on rules in `scoring_rules.yaml`:

| Factor | Examples | Effect |
|--------|----------|--------|
| Extension | `.md` +50, `.pdf` +40, `.py` +5, `.jpg` -100 | Base score |
| Path patterns | `*/docs/*` +20, `*/features/*` +25, `*CLAUDE.md` +20 | Boost/demote |
| Size | 500B-1MB +5, <100B -15, >10MB -25 | Adjust |
| Threshold | Score < 35 | Skipped |

To check how files would score without running a full crawl:

```bash
python -m share_scout crawl --dry-run
```

## Web UI Pages

### Dashboard

Overview of the catalog: file counts by status, score distribution, category breakdown, extension counts, project-level stats, LLM throughput metrics, and Ollama connection status.

### Browse

Filterable file list. Filter by category, extension, project, status, or score range. Sort by any column. Projects are grouped according to `project_groups.yaml`.

### Search

Full-text search across all summaries, keywords, and text samples. Uses SQLite FTS5 — supports standard FTS query syntax like quoted phrases and `OR`.

### Compare

Side-by-side comparison of 2+ projects. Select projects and see their files organized in a grid by category, revealing what document types each project has.

### Tags

Keyword explorer. Browse all keywords extracted by the LLM (filtered to keywords appearing 2+ times). Click a keyword to see all files tagged with it.

### Timeline

Chronological view of analyzed files grouped by month. Filter by project to see when work happened.

### Insights

Actionable intelligence derived from catalog data:

- **Abandoned & Stale Work** — High-scoring files older than 3 months, feature specs with no corresponding code, session handoffs with unfinished next steps.
- **Cross-Project Patterns** — Filenames shared across 3+ projects, technology keyword heatmap across project groups, category balance per group.
- **High-Value Discoveries** — Keyword clusters (files sharing 3+ keywords), unique vs universal keywords across projects.
- **Improvement Opportunities** — Projects missing test/architecture/README files, content-rich files that were under-tagged by the LLM.

### File Detail

Click any file to see its full metadata, LLM summary, keywords, extracted text sample, and related files (by filename and keyword overlap).

## Configuration

### config.yaml

| Setting | Default | Purpose |
|---------|---------|---------|
| `crawl.root_path` | `.` | Directory to crawl |
| `crawl.batch_size` | 100 | Files per DB commit |
| `crawl.skip_dirs` | node_modules, .venv, .git, ... | Directories to skip |
| `crawl.text_sample_max_chars` | 4000 | Max text extracted per file |
| `ollama.endpoint` | `http://localhost:11434` | Ollama API URL |
| `ollama.model` | `mistral:7b` | Model for analysis |
| `ollama.timeout` | 120 | Seconds per LLM call |
| `openai.base_url` | (commented out) | OpenAI-compatible API URL |
| `openai.model` | (commented out) | Model name |
| `openai.api_key_env` | `OPENAI_API_KEY` | Env var containing API key |
| `catalog.db_path` | `share_scout.db` | SQLite database file |
| `web.host` | `0.0.0.0` | Web server bind address |
| `web.port` | 8080 | Web server port |

### scoring_rules.yaml

Controls which files get analyzed. Edit to adjust extension scores, add path boost/demote patterns, change size rules, or adjust the score threshold.

### project_groups.yaml

Groups projects into logical categories for the Browse sidebar and Insights views. Projects not listed here appear under "Other". Add new projects as the crawl discovers them.

## Data

- **Database:** `share_scout.db` (SQLite with WAL mode)
- **Tables:** `files` (all discovered files), `analyses` (LLM results), `crawl_runs` (run history), `analyses_fts` (full-text search index)
- **Resume:** Re-running a crawl skips already-processed files automatically. Only new files get scanned and analyzed.

## CLI Reference

```
python -m share_scout [--config CONFIG] [--rules RULES] [-v] {crawl,web}

crawl options:
  --root-path PATH        Override crawl root directory
  --dry-run               Score only, no extraction or LLM
  --ollama-endpoint URL   Override Ollama API endpoint
  --ollama-model MODEL    Override Ollama model
  --db-path PATH          Override database path
  --batch-size N          Override batch commit size

web options:
  --host HOST             Override web server host
  --port PORT             Override web server port
  --db-path PATH          Override database path
```
