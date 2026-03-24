"""ShareScout MCP server — exposes knowledge base search and RAG query via stdio."""

import json
import logging
import os
import sys

from mcp.server.fastmcp import FastMCP

# Ensure the project root is importable and config loads from the right place
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

from share_scout.catalog import Catalog
from share_scout.config import load_config
from share_scout.rag import ask as rag_ask

logger = logging.getLogger(__name__)

mcp = FastMCP("sharescout", instructions="Search and query a local knowledge base of crawled documents.")

# Global state — initialized on first use
_config = None
_catalog = None


def _init():
    """Lazy-init config and catalog."""
    global _config, _catalog
    if _catalog is not None:
        return
    _config = load_config(os.path.join(PROJECT_ROOT, "config.yaml"))
    _catalog = Catalog(db_path=_config["catalog"]["db_path"])
    _catalog.connect()


@mcp.tool()
def search(query: str, limit: int = 10) -> str:
    """Full-text search over file summaries and keywords.

    Args:
        query: Search terms (supports FTS5 syntax: AND, OR, NOT, quotes for phrases)
        limit: Maximum number of results (default 10)
    """
    _init()
    results, total = _catalog.search(query, limit=limit)
    output = []
    for r in results:
        keywords = r.get("keywords", [])
        if isinstance(keywords, str):
            keywords = json.loads(keywords)
        output.append({
            "file_id": r["id"],
            "filename": r["filename"],
            "path": r["path"],
            "category": r.get("category"),
            "summary": r.get("summary", ""),
            "keywords": keywords,
            "relevance_score": r.get("relevance_score"),
        })
    return json.dumps({"total": total, "results": output}, indent=2)


@mcp.tool()
def ask(question: str, mode: str = "query") -> str:
    """Ask a question about documents in the knowledge base using RAG.

    Uses semantic search to find relevant documents, then generates an answer.
    Query mode returns extracted sentences with sources. Chat mode allows freeform answers.

    Args:
        question: The question to ask
        mode: "query" for strict extractive answers, "chat" for freeform (default: "query")
    """
    _init()
    if mode not in ("query", "chat"):
        mode = "query"
    result = rag_ask(_config, _catalog, question, mode=mode)
    sources = [
        {"file_id": s["file_id"], "filename": s["filename"], "path": s["path"]}
        for s in result.get("sources", [])
    ]
    return json.dumps({
        "answer": result["answer"],
        "sources": sources,
    }, indent=2)


@mcp.tool()
def get_file(file_id: int) -> str:
    """Get full details and analysis for a specific file by its ID.

    Args:
        file_id: The file ID from search results
    """
    _init()
    detail = _catalog.get_file_detail(file_id)
    if not detail:
        return json.dumps({"error": f"File {file_id} not found"})
    analysis = detail.get("analysis")
    output = {
        "file_id": detail["id"],
        "filename": detail["filename"],
        "path": detail["path"],
        "extension": detail.get("extension"),
        "size_bytes": detail.get("size_bytes"),
        "modified_at": detail.get("modified_at"),
        "relevance_score": detail.get("relevance_score"),
        "status": detail.get("status"),
    }
    if analysis:
        keywords = analysis.get("keywords", [])
        if isinstance(keywords, str):
            keywords = json.loads(keywords)
        output["analysis"] = {
            "summary": analysis.get("summary", ""),
            "keywords": keywords,
            "category": analysis.get("category"),
        }
    return json.dumps(output, indent=2)


@mcp.tool()
def list_categories() -> str:
    """List all document categories with file counts."""
    _init()
    categories = _catalog.get_categories()
    return json.dumps(categories, indent=2)


def main():
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
