"""QA test scenarios for ShareScout.

Each scenario defines what to test, how to test it, and what success looks like.
The Test Agent executes the scenario, the Judge Agent scores the results.
"""

SCENARIOS = {
    # ── Tier 1: Single component, isolated ──
    "crawl-dry-run": {
        "id": "crawl-dry-run",
        "name": "Dry run crawl produces valid output",
        "tier": 1,
        "task": (
            "Run a dry-run crawl on the ShareScout project directory itself.\n"
            "Command: `python -m share_scout crawl --root-path /home/neil/dev/kbase --dry-run`\n"
            "Capture the full output. Verify it shows file counts, score distribution, "
            "and top extensions. The output should show Total files > 0, "
            "Would analyze > 0, and at least .py and .html in top extensions."
        ),
        "success_criteria": {
            "required": [
                "Dry run command exits successfully (exit code 0)",
                "Output includes 'Total files found' with a count > 0",
                "Output includes 'Would analyze' with a count > 0",
                "Output includes 'Score distribution' section",
                "Output includes 'Top extensions' section with .py listed",
            ],
            "bonus": [
                ".html, .yaml, and .md also appear in top extensions",
                "Would skip count > 0 (some files are below threshold)",
            ],
        },
    },
    "catalog-schema": {
        "id": "catalog-schema",
        "name": "Catalog schema initializes correctly",
        "tier": 1,
        "task": (
            "Test that the catalog schema creates all required tables including the new "
            "vector/embedding tables. Run this Python snippet and capture output:\n"
            "```python\n"
            "from share_scout.catalog import Catalog\n"
            "cat = Catalog(':memory:')\n"
            "with cat.connection():\n"
            "    cat.init_schema()\n"
            "    tables = [r[0] for r in cat._conn.execute(\n"
            "        \"SELECT name FROM sqlite_master WHERE type='table' ORDER BY name\"\n"
            "    ).fetchall()]\n"
            "    print('Tables:', tables)\n"
            "    print('has_vec:', cat._has_vec)\n"
            "    stats = cat.get_embedding_stats()\n"
            "    print('Embedding stats:', stats)\n"
            "    print('Has embeddings:', cat.has_embeddings())\n"
            "```\n"
        ),
        "success_criteria": {
            "required": [
                "Script runs without errors",
                "Tables include: files, analyses, chunk_summaries, embeddings_meta, settings, crawl_runs",
                "has_vec is True (sqlite-vec loaded)",
                "get_embedding_stats returns a dict with vec_available=True",
                "has_embeddings returns False (empty DB)",
            ],
            "bonus": [
                "analyses_fts virtual table exists",
                "chunks_vec virtual table exists",
            ],
        },
    },
    "embed-model-check": {
        "id": "embed-model-check",
        "name": "Embedding model check and generation",
        "tier": 1,
        "task": (
            "Test that the embedding pipeline can reach the model and generate a vector.\n"
            "```python\n"
            "from share_scout.llm_client import check_embedding_model, generate_embedding\n"
            "from share_scout.config import load_config\n"
            "config = load_config('config.yaml')\n"
            "print('Model check:', check_embedding_model(config))\n"
            "vec = generate_embedding(config, 'test document about software architecture')\n"
            "if vec:\n"
            "    print('Vector dims:', len(vec))\n"
            "    print('First 3 values:', [round(v, 4) for v in vec[:3]])\n"
            "else:\n"
            "    print('ERROR: embedding generation failed')\n"
            "```\n"
            "This requires the Ollama tunnel to be active with nomic-embed-text available."
        ),
        "success_criteria": {
            "required": [
                "check_embedding_model returns True",
                "generate_embedding returns a non-None vector",
                "Vector has exactly 768 dimensions",
                "Vector values are valid floats (not NaN or Inf)",
            ],
        },
    },

    # ── Tier 2: Multi-component integration ──
    "embed-and-search": {
        "id": "embed-and-search",
        "name": "Embed documents and vector search returns results",
        "tier": 2,
        "task": (
            "Test the full embed → search pipeline using an in-memory catalog.\n"
            "```python\n"
            "from share_scout.catalog import Catalog\n"
            "from share_scout.llm_client import generate_embedding\n"
            "from share_scout.config import load_config\n"
            "config = load_config('config.yaml')\n"
            "cat = Catalog(':memory:')\n"
            "with cat.connection():\n"
            "    cat.init_schema()\n"
            "    # Insert 3 test files with different topics\n"
            "    docs = [\n"
            "        ('auth.py', 'Code for user authentication, login, password hashing, JWT tokens'),\n"
            "        ('deploy.md', 'Deployment guide for AWS EC2, Docker containers, nginx reverse proxy'),\n"
            "        ('api.py', 'REST API endpoints for user management, CRUD operations, pagination'),\n"
            "    ]\n"
            "    for i, (name, text) in enumerate(docs, 1):\n"
            "        cat._conn.execute(\n"
            "            'INSERT INTO files (id, path, filename, extension, status, relevance_score) '\n"
            "            'VALUES (?, ?, ?, ?, ?, ?)',\n"
            "            (i, f'/test/{name}', name, name.split('.')[-1], 'analyzed', 50)\n"
            "        )\n"
            "        cat._conn.execute(\n"
            "            'INSERT INTO analyses (file_id, text_sample, summary, keywords, category) '\n"
            "            'VALUES (?, ?, ?, ?, ?)',\n"
            "            (i, text, f'Summary of {name}', '[]', 'Test')\n"
            "        )\n"
            "        vec = generate_embedding(config, text)\n"
            "        if vec:\n"
            "            cat.insert_embedding(i, None, 'text_sample', vec, 'nomic-embed-text')\n"
            "    cat.commit()\n"
            "    # Search for auth-related content\n"
            "    query_vec = generate_embedding(config, 'How does authentication work?')\n"
            "    results = cat.vector_search(query_vec, limit=3)\n"
            "    print(f'Search returned {len(results)} results')\n"
            "    for r in results:\n"
            "        print(f'  dist={r[\"distance\"]:.3f} file={r[\"filename\"]}')\n"
            "    print(f'Top result: {results[0][\"filename\"] if results else \"NONE\"}')\n"
            "    stats = cat.get_embedding_stats()\n"
            "    print(f'Stats: {stats}')\n"
            "```\n"
        ),
        "success_criteria": {
            "required": [
                "All 3 documents embedded without error",
                "Vector search returns 3 results",
                "auth.py is the top result for 'How does authentication work?'",
                "Embedding stats show 3 files with embeddings",
            ],
            "bonus": [
                "deploy.md is NOT the top result (semantic relevance works)",
                "Distance for auth.py is meaningfully lower than for deploy.md",
            ],
        },
    },
    "rag-grounded-answer": {
        "id": "rag-grounded-answer",
        "name": "RAG answer is grounded in sources, not hallucinated",
        "tier": 2,
        "task": (
            "Test that the RAG pipeline produces an answer grounded in the provided documents.\n"
            "Use the live catalog (share_scout.db) if it has embeddings, otherwise report BLOCKED.\n"
            "```python\n"
            "from share_scout.catalog import Catalog\n"
            "from share_scout.rag import ask\n"
            "from share_scout.config import load_config\n"
            "config = load_config('config.yaml')\n"
            "cat = Catalog('share_scout.db')\n"
            "with cat.connection():\n"
            "    cat.init_schema()\n"
            "    if not cat.has_embeddings():\n"
            "        print('BLOCKED: no embeddings in catalog')\n"
            "    else:\n"
            "        result = ask(config, cat, 'What is ShareScout and what does it do?', top_k=3)\n"
            "        print('Answer:', result['answer'][:500])\n"
            "        print(f'Sources: {len(result[\"sources\"])}')\n"
            "        for s in result['sources']:\n"
            "            print(f'  {s[\"filename\"]} dist={s[\"distance\"]:.3f}')\n"
            "```\n"
            "The answer should reference actual document content, not generic LLM knowledge."
        ),
        "success_criteria": {
            "required": [
                "RAG returns a non-empty answer",
                "At least 1 source document is returned",
                "Answer mentions specific details from the source documents (not generic knowledge)",
                "No hallucinated project names or features not in the excerpts",
            ],
            "bonus": [
                "Answer cites source filenames as instructed by the system prompt",
                "Sources are from relevant ShareScout docs (README, CLAUDE.md, etc.)",
            ],
        },
    },

    # ── Tier 3: Edge cases and error handling ──
    "embed-no-model": {
        "id": "embed-no-model",
        "name": "Embed command fails gracefully without model",
        "tier": 3,
        "task": (
            "Test that the embed command handles missing/unreachable model gracefully.\n"
            "```python\n"
            "from share_scout.embedder import run_embed\n"
            "import logging\n"
            "logging.basicConfig(level=logging.INFO)\n"
            "# Config with non-existent model\n"
            "config = {\n"
            "    'ollama': {'embedding_model': 'nonexistent-model-xyz', 'endpoint': 'http://localhost:11434'},\n"
            "    'catalog': {'db_path': ':memory:'},\n"
            "    'crawl': {'batch_size': 10},\n"
            "}\n"
            "run_embed(config)  # Should log error and return, not crash\n"
            "print('Completed without crash')\n"
            "```\n"
        ),
        "success_criteria": {
            "required": [
                "run_embed does not raise an exception",
                "An error message is logged about the model not being available",
                "Script prints 'Completed without crash'",
            ],
        },
    },
    "ask-no-embeddings": {
        "id": "ask-no-embeddings",
        "name": "Ask route handles empty catalog gracefully",
        "tier": 3,
        "task": (
            "Test the /ask web route when no embeddings exist.\n"
            "```python\n"
            "from share_scout.web.app import create_app\n"
            "config = {\n"
            "    'crawl': {'root_path': '/tmp', 'batch_size': 100},\n"
            "    'catalog': {'db_path': ':memory:'},\n"
            "    'web': {'host': '0.0.0.0', 'port': 8080},\n"
            "    'ollama': {'endpoint': 'http://localhost:11434', 'model': 'test'},\n"
            "}\n"
            "app = create_app(config)\n"
            "client = app.test_client()\n"
            "# GET /ask should show 'no embeddings' message\n"
            "resp = client.get('/ask')\n"
            "print(f'GET /ask: {resp.status_code}')\n"
            "print(f'Contains no-embeddings warning: {b\"embed\" in resp.data}')\n"
            "# POST /ask should not crash\n"
            "resp2 = client.post('/ask', data={'question': 'test question'})\n"
            "print(f'POST /ask: {resp2.status_code}')\n"
            "```\n"
        ),
        "success_criteria": {
            "required": [
                "GET /ask returns 200",
                "Response contains the no-embeddings warning message",
                "POST /ask does not crash (returns 200 or redirect)",
            ],
        },
    },
    "web-routes-smoke": {
        "id": "web-routes-smoke",
        "name": "All web routes return 200",
        "tier": 3,
        "task": (
            "Test that all web routes respond without errors using the test client.\n"
            "```python\n"
            "from share_scout.web.app import create_app\n"
            "config = {\n"
            "    'crawl': {'root_path': '/tmp', 'batch_size': 100},\n"
            "    'catalog': {'db_path': 'share_scout.db'},\n"
            "    'web': {'host': '0.0.0.0', 'port': 8080},\n"
            "    'ollama': {'endpoint': 'http://localhost:11434', 'model': 'test',\n"
            "              'embedding_model': 'nomic-embed-text'},\n"
            "}\n"
            "app = create_app(config)\n"
            "client = app.test_client()\n"
            "routes = ['/', '/browse', '/search', '/ask', '/compare', '/tags',\n"
            "          '/timeline', '/insights', '/settings', '/crawl/status']\n"
            "for route in routes:\n"
            "    resp = client.get(route)\n"
            "    status = 'OK' if resp.status_code == 200 else f'FAIL ({resp.status_code})'\n"
            "    print(f'{route:20s} {status}')\n"
            "```\n"
        ),
        "success_criteria": {
            "required": [
                "All routes return HTTP 200",
                "No Python exceptions or tracebacks in output",
                "Dashboard (/) renders successfully",
                "/ask route is accessible",
            ],
            "bonus": [
                "/crawl/status returns valid JSON",
                "/search works with an empty query",
            ],
        },
    },

    # ── Tier 4: End-to-end realistic workflow ──
    "full-pipeline": {
        "id": "full-pipeline",
        "name": "Full pipeline: crawl → analyze → embed → ask",
        "tier": 4,
        "task": (
            "Run the complete ShareScout pipeline on a small test directory.\n\n"
            "1. Create a temp directory with 3 test files:\n"
            "   - `auth-guide.md`: 'Authentication uses JWT tokens. Users login with email/password. "
            "Tokens expire after 24 hours. Refresh tokens stored in httpOnly cookies.'\n"
            "   - `deploy-notes.md`: 'Deploy to AWS using Docker. Run docker-compose up. "
            "Nginx handles TLS termination. Health check at /api/health.'\n"
            "   - `api-reference.md`: 'GET /users returns paginated list. POST /users creates new user. "
            "PUT /users/:id updates user. DELETE /users/:id soft-deletes.'\n\n"
            "2. Run a crawl on that directory (NOT dry-run) — this needs an LLM endpoint\n"
            "3. Run embed on the resulting catalog\n"
            "4. Ask a question: 'How do users authenticate?'\n"
            "5. Verify the answer references JWT tokens and the auth-guide.md source\n\n"
            "Use a temp DB path to avoid polluting the main catalog.\n"
            "If the LLM or embedding endpoint is unreachable, report BLOCKED."
        ),
        "success_criteria": {
            "required": [
                "Test files created in temp directory",
                "Crawl completes without errors",
                "At least 2 of 3 files are analyzed",
                "Embed produces vectors for analyzed files",
                "RAG question returns an answer",
                "Answer mentions JWT tokens or authentication details from the test file",
                "auth-guide.md appears in the sources",
            ],
            "bonus": [
                "All 3 files analyzed",
                "auth-guide.md is the top-ranked source",
                "Answer does not contain information not in the test files",
                "Cleanup: temp directory and DB removed after test",
            ],
        },
    },
}

# Keyword shortcuts for quick scenario selection
SCENARIO_KEYWORDS = {
    "crawl": "crawl-dry-run",
    "dry-run": "crawl-dry-run",
    "schema": "catalog-schema",
    "catalog": "catalog-schema",
    "model": "embed-model-check",
    "embedding": "embed-model-check",
    "search": "embed-and-search",
    "vector": "embed-and-search",
    "rag": "rag-grounded-answer",
    "grounded": "rag-grounded-answer",
    "no-model": "embed-no-model",
    "graceful": "embed-no-model",
    "no-embed": "ask-no-embeddings",
    "routes": "web-routes-smoke",
    "web": "web-routes-smoke",
    "smoke": "web-routes-smoke",
    "pipeline": "full-pipeline",
    "e2e": "full-pipeline",
    "full": "full-pipeline",
}


def get_scenario(keyword: str) -> dict | None:
    """Look up a scenario by ID or keyword."""
    if keyword in SCENARIOS:
        return SCENARIOS[keyword]
    scenario_id = SCENARIO_KEYWORDS.get(keyword)
    if scenario_id:
        return SCENARIOS.get(scenario_id)
    return None


def list_scenarios() -> str:
    """Format scenarios for display."""
    lines = []
    for tier in [1, 2, 3, 4]:
        tier_scenarios = [s for s in SCENARIOS.values() if s["tier"] == tier]
        if tier_scenarios:
            lines.append(f"\n## Tier {tier}")
            for s in tier_scenarios:
                req_count = len(s["success_criteria"]["required"])
                lines.append(f"  **{s['id']}** — {s['name']} ({req_count} required criteria)")
    lines.append("\n## Keywords")
    for kw, sid in sorted(SCENARIO_KEYWORDS.items()):
        lines.append(f"  {kw} → {sid}")
    return "\n".join(lines)
