"""SQLite catalog: schema creation, CRUD operations, queries, FTS5."""

import json
import re
import sqlite3
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    filename TEXT NOT NULL,
    extension TEXT,
    size_bytes INTEGER,
    modified_at TIMESTAMP,
    content_hash TEXT,
    relevance_score REAL,
    status TEXT DEFAULT 'pending',
    skip_reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY,
    file_id INTEGER REFERENCES files(id),
    text_sample TEXT,
    summary TEXT,
    keywords TEXT,
    category TEXT,
    llm_stats TEXT,
    analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS crawl_runs (
    id INTEGER PRIMARY KEY,
    root_path TEXT NOT NULL,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    files_found INTEGER DEFAULT 0,
    files_analyzed INTEGER DEFAULT 0,
    status TEXT DEFAULT 'running'
);

CREATE INDEX IF NOT EXISTS idx_files_extension ON files(extension);
CREATE INDEX IF NOT EXISTS idx_files_score ON files(relevance_score);
CREATE INDEX IF NOT EXISTS idx_files_status ON files(status);
CREATE INDEX IF NOT EXISTS idx_analyses_category ON analyses(category);
CREATE INDEX IF NOT EXISTS idx_analyses_file_id ON analyses(file_id);
"""

FTS_SCHEMA_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS analyses_fts USING fts5(
    summary, keywords, text_sample, content='analyses', content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS analyses_ai AFTER INSERT ON analyses BEGIN
    INSERT INTO analyses_fts(rowid, summary, keywords, text_sample)
    VALUES (new.id, new.summary, new.keywords, new.text_sample);
END;

CREATE TRIGGER IF NOT EXISTS analyses_ad AFTER DELETE ON analyses BEGIN
    INSERT INTO analyses_fts(analyses_fts, rowid, summary, keywords, text_sample)
    VALUES ('delete', old.id, old.summary, old.keywords, old.text_sample);
END;

CREATE TRIGGER IF NOT EXISTS analyses_au AFTER UPDATE ON analyses BEGIN
    INSERT INTO analyses_fts(analyses_fts, rowid, summary, keywords, text_sample)
    VALUES ('delete', old.id, old.summary, old.keywords, old.text_sample);
    INSERT INTO analyses_fts(rowid, summary, keywords, text_sample)
    VALUES (new.id, new.summary, new.keywords, new.text_sample);
END;
"""


class Catalog:
    def __init__(self, db_path: str = "share_scout.db", root_path: str = None):
        self.db_path = db_path
        self._root_path = root_path
        self._conn = None

    def connect(self):
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    @contextmanager
    def connection(self):
        self.connect()
        try:
            yield self
        finally:
            self.close()

    def init_schema(self):
        self._conn.executescript(SCHEMA_SQL)
        self._conn.executescript(FTS_SCHEMA_SQL)
        self._conn.commit()

    # -- Crawl runs --

    def start_crawl_run(self, root_path: str) -> int:
        cur = self._conn.execute(
            "INSERT INTO crawl_runs (root_path, started_at, status) VALUES (?, ?, 'running')",
            (root_path, datetime.now().isoformat()),
        )
        self._conn.commit()
        return cur.lastrowid

    def update_crawl_run(self, run_id: int, **kwargs):
        sets = []
        vals = []
        for k, v in kwargs.items():
            sets.append(f"{k} = ?")
            vals.append(v)
        vals.append(run_id)
        self._conn.execute(
            f"UPDATE crawl_runs SET {', '.join(sets)} WHERE id = ?", vals
        )
        self._conn.commit()

    def complete_crawl_run(self, run_id: int, files_found: int, files_analyzed: int):
        self._conn.execute(
            "UPDATE crawl_runs SET completed_at = ?, files_found = ?, files_analyzed = ?, status = 'completed' WHERE id = ?",
            (datetime.now().isoformat(), files_found, files_analyzed, run_id),
        )
        self._conn.commit()

    def get_latest_crawl_run(self) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM crawl_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None

    # -- Files --

    def upsert_file(self, file_data: dict):
        self._conn.execute(
            """INSERT INTO files (path, filename, extension, size_bytes, modified_at,
                                  relevance_score, status, skip_reason, content_hash, updated_at)
               VALUES (:path, :filename, :extension, :size_bytes, :modified_at,
                       :relevance_score, :status, :skip_reason, :content_hash, :updated_at)
               ON CONFLICT(path) DO UPDATE SET
                   size_bytes=excluded.size_bytes, modified_at=excluded.modified_at,
                   relevance_score=excluded.relevance_score, status=excluded.status,
                   skip_reason=excluded.skip_reason, content_hash=excluded.content_hash,
                   updated_at=excluded.updated_at""",
            {
                "path": file_data["path"],
                "filename": file_data["filename"],
                "extension": file_data.get("extension"),
                "size_bytes": file_data.get("size_bytes"),
                "modified_at": file_data.get("modified_at"),
                "relevance_score": file_data.get("relevance_score"),
                "status": file_data.get("status", "pending"),
                "skip_reason": file_data.get("skip_reason"),
                "content_hash": file_data.get("content_hash"),
                "updated_at": datetime.now().isoformat(),
            },
        )

    def upsert_files_batch(self, files: list[dict]):
        for f in files:
            self.upsert_file(f)
        self._conn.commit()

    def get_file(self, file_id: int) -> dict | None:
        row = self._conn.execute("SELECT * FROM files WHERE id = ?", (file_id,)).fetchone()
        return dict(row) if row else None

    def get_file_by_path(self, path: str) -> dict | None:
        row = self._conn.execute("SELECT * FROM files WHERE path = ?", (path,)).fetchone()
        return dict(row) if row else None

    def file_exists(self, path: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM files WHERE path = ? AND status != 'pending'", (path,)
        ).fetchone()
        return row is not None

    def get_pending_files(self, limit: int = 100) -> list[dict]:
        """Get files awaiting LLM analysis, highest score first."""
        rows = self._conn.execute(
            "SELECT * FROM files WHERE status = 'extracted' ORDER BY relevance_score DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    # -- Analyses --

    def insert_analysis(self, file_id: int, text_sample: str, summary: str,
                        keywords: list[str], category: str, llm_stats: dict = None):
        # Delete existing analysis for this file (re-crawl)
        self._conn.execute("DELETE FROM analyses WHERE file_id = ?", (file_id,))
        self._conn.execute(
            """INSERT INTO analyses (file_id, text_sample, summary, keywords, category, llm_stats, analyzed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (file_id, text_sample, summary, json.dumps(keywords), category,
             json.dumps(llm_stats) if llm_stats else None,
             datetime.now().isoformat()),
        )

    def get_analysis(self, file_id: int) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM analyses WHERE file_id = ?", (file_id,)
        ).fetchone()
        if row:
            d = dict(row)
            d["keywords"] = json.loads(d["keywords"]) if d["keywords"] else []
            return d
        return None

    def commit(self):
        self._conn.commit()

    # -- Query helpers for web UI --

    def get_stats(self) -> dict:
        stats = {}
        row = self._conn.execute("SELECT COUNT(*) as cnt FROM files").fetchone()
        stats["total_files"] = row["cnt"]
        row = self._conn.execute("SELECT COUNT(*) as cnt FROM files WHERE status = 'analyzed'").fetchone()
        stats["analyzed"] = row["cnt"]
        row = self._conn.execute("SELECT COUNT(*) as cnt FROM files WHERE status = 'skipped'").fetchone()
        stats["skipped"] = row["cnt"]
        row = self._conn.execute("SELECT COUNT(*) as cnt FROM files WHERE status = 'extracted'").fetchone()
        stats["extracted"] = row["cnt"]
        row = self._conn.execute("SELECT COUNT(*) as cnt FROM files WHERE status = 'pending'").fetchone()
        stats["pending"] = row["cnt"]
        row = self._conn.execute("SELECT COALESCE(SUM(size_bytes), 0) as total FROM files").fetchone()
        stats["total_size_bytes"] = row["total"]
        stats["latest_crawl"] = self.get_latest_crawl_run()
        return stats

    def get_all_crawl_runs(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM crawl_runs ORDER BY id DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_recent_analyses(self, limit: int = 20) -> list[dict]:
        rows = self._conn.execute("""
            SELECT f.filename, f.path, f.relevance_score, f.id as file_id,
                   a.summary, a.category, a.analyzed_at
            FROM analyses a
            JOIN files f ON f.id = a.file_id
            ORDER BY a.analyzed_at DESC
            LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in rows]

    def get_projects(self) -> list[dict]:
        """Get distinct project names with file counts."""
        rows = self._conn.execute(
            "SELECT path FROM files WHERE status = 'analyzed'"
        ).fetchall()
        from collections import Counter
        counter = Counter()
        for r in rows:
            project = self._extract_project(r["path"], self._root_path)
            counter[project] += 1
        return sorted(
            [{"project": k, "count": v} for k, v in counter.items()],
            key=lambda x: -x["count"]
        )

    def get_project_stats(self) -> list[dict]:
        """Get file counts and analysis stats grouped by project (first dir under root)."""
        rows = self._conn.execute(
            "SELECT path, status, size_bytes FROM files"
        ).fetchall()
        from collections import defaultdict
        stats = defaultdict(lambda: {"project": "", "total": 0, "analyzed": 0,
                                      "extracted": 0, "skipped": 0, "total_size": 0})
        for r in rows:
            project = self._extract_project(r["path"], self._root_path)
            s = stats[project]
            s["project"] = project
            s["total"] += 1
            s["total_size"] += r["size_bytes"] or 0
            status = r["status"]
            if status == "analyzed":
                s["analyzed"] += 1
            elif status == "extracted":
                s["extracted"] += 1
            elif status == "skipped":
                s["skipped"] += 1
        return sorted(stats.values(), key=lambda x: -x["total"])

    def get_analysis_rate(self) -> dict:
        """Get analysis throughput stats including LLM token/timing aggregates."""
        rate = {"per_minute": 0, "avg_seconds": 0, "total_analyses": 0,
                "first_at": None, "last_at": None,
                "total_tokens": 0, "total_prompt_tokens": 0,
                "total_eval_ms": 0, "total_llm_ms": 0,
                "avg_tokens_per_file": 0, "tokens_per_second": 0}
        row = self._conn.execute("SELECT COUNT(*) as cnt FROM analyses").fetchone()
        rate["total_analyses"] = row["cnt"]
        if rate["total_analyses"] < 2:
            return rate
        row = self._conn.execute(
            "SELECT MIN(analyzed_at) as first_at, MAX(analyzed_at) as last_at FROM analyses"
        ).fetchone()
        rate["first_at"] = row["first_at"]
        rate["last_at"] = row["last_at"]
        if row["first_at"] and row["last_at"]:
            from datetime import datetime
            first = datetime.fromisoformat(row["first_at"])
            last = datetime.fromisoformat(row["last_at"])
            elapsed = (last - first).total_seconds()
            if elapsed > 0:
                rate["per_minute"] = round(rate["total_analyses"] / (elapsed / 60), 1)
                rate["avg_seconds"] = round(elapsed / rate["total_analyses"], 1)
                # Estimate remaining time for extracted files
                extracted = self._conn.execute(
                    "SELECT COUNT(*) as cnt FROM files WHERE status = 'extracted'"
                ).fetchone()["cnt"]
                hours_left = 0
                if rate["per_minute"] > 0:
                    hours_left = round(extracted / rate["per_minute"] / 60, 1)
                rate["remaining_files"] = extracted
                rate["eta_hours"] = hours_left

        # Aggregate LLM stats from analyses that have them
        rows = self._conn.execute(
            "SELECT llm_stats FROM analyses WHERE llm_stats IS NOT NULL"
        ).fetchall()
        total_eval_tokens = 0
        total_prompt_tokens = 0
        total_eval_ms = 0
        total_llm_ms = 0
        stats_count = 0
        for row in rows:
            try:
                s = json.loads(row["llm_stats"])
                total_eval_tokens += s.get("eval_count", 0)
                total_prompt_tokens += s.get("prompt_eval_count", 0)
                total_eval_ms += s.get("eval_duration_ms", 0)
                total_llm_ms += s.get("total_duration_ms", 0)
                stats_count += 1
            except (json.JSONDecodeError, TypeError):
                continue

        rate["total_tokens"] = total_eval_tokens + total_prompt_tokens
        rate["total_prompt_tokens"] = total_prompt_tokens
        rate["total_eval_tokens"] = total_eval_tokens
        rate["total_eval_ms"] = total_eval_ms
        rate["total_llm_ms"] = total_llm_ms
        rate["stats_count"] = stats_count
        if stats_count > 0:
            rate["avg_tokens_per_file"] = round(rate["total_tokens"] / stats_count)
            if total_eval_ms > 0:
                rate["tokens_per_second"] = round(total_eval_tokens / (total_eval_ms / 1000), 1)
            rate["avg_llm_ms"] = round(total_llm_ms / stats_count)

        return rate

    def get_live_status(self) -> dict:
        """Get the most recently processed file, last analyzed, and next in queue."""
        status = {}
        # Last file touched (any status)
        row = self._conn.execute("""
            SELECT f.filename, f.path, f.status, f.relevance_score, f.updated_at,
                   a.category, a.summary
            FROM files f LEFT JOIN analyses a ON f.id = a.file_id
            ORDER BY f.updated_at DESC LIMIT 1
        """).fetchone()
        status["last_processed"] = dict(row) if row else None

        # Last analyzed
        row = self._conn.execute("""
            SELECT f.filename, f.path, f.relevance_score, f.id as file_id,
                   a.category, a.summary, a.analyzed_at
            FROM analyses a JOIN files f ON f.id = a.file_id
            ORDER BY a.analyzed_at DESC LIMIT 1
        """).fetchone()
        status["last_analyzed"] = dict(row) if row else None

        # Next in queue (highest-scoring extracted file)
        row = self._conn.execute("""
            SELECT filename, path, relevance_score, extension
            FROM files
            WHERE status = 'extracted'
            ORDER BY relevance_score DESC LIMIT 1
        """).fetchone()
        status["next_in_queue"] = dict(row) if row else None

        # Queue depth
        row = self._conn.execute(
            "SELECT COUNT(*) as cnt FROM files WHERE status = 'extracted'"
        ).fetchone()
        status["queue_depth"] = row["cnt"]

        # Crawl active?
        row = self._conn.execute(
            "SELECT * FROM crawl_runs WHERE status = 'running' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        status["active_crawl"] = dict(row) if row else None

        return status

    def get_similar_files(self, filename: str, limit: int = 5) -> list[dict]:
        """Find already-analyzed files with the same filename for context."""
        rows = self._conn.execute("""
            SELECT f.path, f.filename, a.summary, a.category
            FROM files f
            JOIN analyses a ON f.id = a.file_id
            WHERE f.filename = ?
            ORDER BY a.analyzed_at DESC
            LIMIT ?
        """, (filename, limit)).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            d["project"] = self._extract_project(d["path"], self._root_path)
            results.append(d)
        return results

    def get_recent_errors(self, limit: int = 20) -> list[dict]:
        """Get recently skipped files with reasons."""
        rows = self._conn.execute("""
            SELECT filename, path, relevance_score, skip_reason, updated_at
            FROM files
            WHERE status = 'skipped' AND skip_reason IS NOT NULL
            ORDER BY updated_at DESC
            LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in rows]

    def get_score_distribution(self) -> list[dict]:
        rows = self._conn.execute("""
            SELECT
                CASE
                    WHEN relevance_score < 0 THEN 'negative'
                    WHEN relevance_score < 20 THEN '0-19'
                    WHEN relevance_score < 40 THEN '20-39'
                    WHEN relevance_score < 60 THEN '40-59'
                    WHEN relevance_score < 80 THEN '60-79'
                    ELSE '80-100'
                END as bracket,
                COUNT(*) as count
            FROM files
            GROUP BY bracket
            ORDER BY MIN(relevance_score)
        """).fetchall()
        return [dict(r) for r in rows]

    def get_categories(self) -> list[dict]:
        rows = self._conn.execute("""
            SELECT category, COUNT(*) as count
            FROM analyses
            WHERE category IS NOT NULL
            GROUP BY category
            ORDER BY count DESC
        """).fetchall()
        return [dict(r) for r in rows]

    def get_top_extensions(self, limit: int = 15) -> list[dict]:
        rows = self._conn.execute("""
            SELECT extension, COUNT(*) as count
            FROM files
            WHERE extension IS NOT NULL
            GROUP BY extension
            ORDER BY count DESC
            LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in rows]

    def browse_files(self, category: str = None, extension: str = None,
                     project: str = None, status: str = None,
                     min_score: float = None, max_score: float = None,
                     sort_by: str = "relevance_score", sort_dir: str = "DESC",
                     limit: int = 50, offset: int = 0) -> tuple[list[dict], int]:
        """Browse files with filters. Returns (files, total_count)."""
        where = ["1=1"]
        params = []

        if category:
            where.append("a.category = ?")
            params.append(category)
        if extension:
            where.append("f.extension = ?")
            params.append(extension)
        if project:
            where.append("(f.path LIKE ? OR f.path LIKE ?)")
            params.append(f"%/{project}/%")
            params.append(f"%\\{project}\\%")
        if status:
            where.append("f.status = ?")
            params.append(status)
        if min_score is not None:
            where.append("f.relevance_score >= ?")
            params.append(min_score)
        if max_score is not None:
            where.append("f.relevance_score <= ?")
            params.append(max_score)

        where_clause = " AND ".join(where)

        allowed_sorts = {"relevance_score", "filename", "size_bytes", "modified_at", "path"}
        if sort_by not in allowed_sorts:
            sort_by = "relevance_score"
        sort_dir = "DESC" if sort_dir.upper() == "DESC" else "ASC"

        count_row = self._conn.execute(
            f"SELECT COUNT(*) as cnt FROM files f LEFT JOIN analyses a ON f.id = a.file_id WHERE {where_clause}",
            params,
        ).fetchone()

        rows = self._conn.execute(
            f"""SELECT f.*, a.summary, a.keywords, a.category
                FROM files f LEFT JOIN analyses a ON f.id = a.file_id
                WHERE {where_clause}
                ORDER BY f.{sort_by} {sort_dir}
                LIMIT ? OFFSET ?""",
            params + [limit, offset],
        ).fetchall()

        files = []
        for r in rows:
            d = dict(r)
            if d.get("keywords"):
                d["keywords"] = json.loads(d["keywords"])
            files.append(d)

        return files, count_row["cnt"]

    def search(self, query: str, limit: int = 50, offset: int = 0) -> tuple[list[dict], int]:
        """Full-text search across analyses."""
        count_row = self._conn.execute(
            "SELECT COUNT(*) as cnt FROM analyses_fts WHERE analyses_fts MATCH ?",
            (query,),
        ).fetchone()

        rows = self._conn.execute(
            """SELECT f.*, a.summary, a.keywords, a.category,
                      highlight(analyses_fts, 0, '<mark>', '</mark>') as summary_hl
               FROM analyses_fts
               JOIN analyses a ON a.id = analyses_fts.rowid
               JOIN files f ON f.id = a.file_id
               WHERE analyses_fts MATCH ?
               ORDER BY rank
               LIMIT ? OFFSET ?""",
            (query, limit, offset),
        ).fetchall()

        results = []
        for r in rows:
            d = dict(r)
            if d.get("keywords"):
                d["keywords"] = json.loads(d["keywords"])
            results.append(d)

        return results, count_row["cnt"]

    def get_file_detail(self, file_id: int) -> dict | None:
        """Get file with its analysis for detail view."""
        file = self.get_file(file_id)
        if not file:
            return None
        file["analysis"] = self.get_analysis(file_id)
        return file

    # -- Cross-project compare --

    def get_files_by_projects(self, project_names: list[str]) -> list[dict]:
        """Get analyzed files for multiple projects, with category info."""
        if not project_names:
            return []
        placeholders = ",".join("?" for _ in project_names)
        like_clauses = " OR ".join("f.path LIKE ?" for _ in project_names)
        params = [f"%/{p}/%" for p in project_names]
        rows = self._conn.execute(f"""
            SELECT f.id, f.filename, f.path, f.relevance_score,
                   a.category, a.summary, a.keywords
            FROM files f
            JOIN analyses a ON f.id = a.file_id
            WHERE f.status = 'analyzed' AND ({like_clauses})
            ORDER BY a.category, f.filename
        """, params).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            if d.get("keywords"):
                d["keywords"] = json.loads(d["keywords"])
            d["project"] = self._extract_project(d["path"], self._root_path)
            results.append(d)
        return results

    # -- Keyword/tag explorer --

    @staticmethod
    def _is_valid_keyword(kw: str) -> bool:
        """Filter out junk keywords from LLM output."""
        if not kw or len(kw) < 2 or len(kw) > 60:
            return False
        # Must contain at least one letter
        if not any(c.isalpha() for c in kw):
            return False
        # Skip paths, filenames, URLs
        if any(c in kw for c in ('/', '\\', '{', '}', '(', ')')):
            return False
        if kw.startswith('.') or kw.startswith('-') or kw.startswith('@'):
            return False
        if any(kw.endswith(ext) for ext in ('.md', '.py', '.js', '.ts', '.sh', '.sql', '.json', '.yaml', '.html')):
            return False
        return True

    def get_all_keywords(self) -> list[dict]:
        """Get all keywords with their counts, sorted alphabetically."""
        rows = self._conn.execute("""
            SELECT keywords FROM analyses WHERE keywords IS NOT NULL
        """).fetchall()
        from collections import Counter
        counter = Counter()
        for r in rows:
            try:
                kws = json.loads(r["keywords"])
                for kw in kws:
                    if kw:
                        clean = kw.strip().lower()
                        if self._is_valid_keyword(clean):
                            counter[clean] += 1
            except (json.JSONDecodeError, TypeError):
                continue
        # Only show keywords that appear 2+ times (filters noise from one-off LLM hallucinations)
        return sorted(
            [{"keyword": k, "count": v} for k, v in counter.items() if v >= 2],
            key=lambda x: (-x["count"], x["keyword"])
        )

    def get_files_by_keyword(self, keyword: str) -> list[dict]:
        """Get all files tagged with a specific keyword."""
        rows = self._conn.execute("""
            SELECT f.id, f.filename, f.path, f.relevance_score,
                   a.category, a.summary, a.keywords
            FROM analyses a
            JOIN files f ON f.id = a.file_id
            WHERE a.keywords IS NOT NULL AND f.status = 'analyzed'
            ORDER BY f.relevance_score DESC
        """).fetchall()
        results = []
        kw_lower = keyword.strip().lower()
        for r in rows:
            d = dict(r)
            try:
                kws = json.loads(d.get("keywords", "[]"))
                if any(k.strip().lower() == kw_lower for k in kws):
                    d["keywords"] = kws
                    d["project"] = self._extract_project(d["path"], self._root_path)
                    results.append(d)
            except (json.JSONDecodeError, TypeError):
                continue
        return results

    # -- Timeline view --

    def get_timeline_files(self, project: str = None) -> list[dict]:
        """Get analyzed files ordered by modified_at descending."""
        where = "f.status = 'analyzed' AND f.modified_at IS NOT NULL"
        params = []
        if project:
            where += " AND (f.path LIKE ? OR f.path LIKE ?)"
            params.extend([f"%/{project}/%", f"%\\{project}\\%"])
        rows = self._conn.execute(f"""
            SELECT f.id, f.filename, f.path, f.relevance_score, f.modified_at,
                   a.category, a.summary
            FROM files f
            JOIN analyses a ON f.id = a.file_id
            WHERE {where}
            ORDER BY f.modified_at DESC
        """, params).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            d["project"] = self._extract_project(d["path"], self._root_path)
            results.append(d)
        return results

    # -- Related files --

    def get_related_by_filename(self, file_id: int, filename: str, limit: int = 10) -> list[dict]:
        """Get files with the same filename from other projects."""
        rows = self._conn.execute("""
            SELECT f.id, f.filename, f.path, f.relevance_score,
                   a.category, a.summary
            FROM files f
            JOIN analyses a ON f.id = a.file_id
            WHERE f.filename = ? AND f.id != ? AND f.status = 'analyzed'
            ORDER BY f.relevance_score DESC
            LIMIT ?
        """, (filename, file_id, limit)).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            d["project"] = self._extract_project(d["path"], self._root_path)
            results.append(d)
        return results

    def get_related_by_keywords(self, file_id: int, keywords: list[str], limit: int = 5) -> list[dict]:
        """Get files with overlapping keywords, ranked by overlap count."""
        if not keywords:
            return []
        rows = self._conn.execute("""
            SELECT f.id, f.filename, f.path, f.relevance_score,
                   a.category, a.summary, a.keywords
            FROM analyses a
            JOIN files f ON f.id = a.file_id
            WHERE f.id != ? AND f.status = 'analyzed' AND a.keywords IS NOT NULL
        """, (file_id,)).fetchall()
        kw_set = {k.strip().lower() for k in keywords if k}
        scored = []
        for r in rows:
            d = dict(r)
            try:
                file_kws = json.loads(d.get("keywords", "[]"))
                overlap = len(kw_set & {k.strip().lower() for k in file_kws if k})
                if overlap > 0:
                    d["keywords"] = file_kws
                    d["overlap"] = overlap
                    d["project"] = self._extract_project(d["path"], self._root_path)
                    scored.append(d)
            except (json.JSONDecodeError, TypeError):
                continue
        scored.sort(key=lambda x: x["overlap"], reverse=True)
        return scored[:limit]

    # -- Insights --

    @staticmethod
    def _extract_project(path: str, root_path: str = None) -> str:
        """Extract project name from a file path."""
        path = path.replace("\\", "/")
        if root_path:
            root = root_path.replace("\\", "/").rstrip("/") + "/"
            if path.startswith(root):
                rel = path[len(root):]
                parts = rel.split("/")
                return parts[0] if parts else "(root)"
        # Fallback: first directory component after any "dev" segment
        parts = path.split("/")
        for i, part in enumerate(parts):
            if part == "dev" and i + 1 < len(parts):
                return parts[i + 1]
        return "(root)"

    def _get_analyzed_files_with_analysis(self) -> list[dict]:
        """Fetch all analyzed files joined with their analysis data."""
        rows = self._conn.execute("""
            SELECT f.id, f.filename, f.path, f.relevance_score, f.modified_at,
                   f.size_bytes, a.summary, a.keywords, a.category, a.text_sample
            FROM files f
            JOIN analyses a ON f.id = a.file_id
            WHERE f.status = 'analyzed'
        """).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            d["project"] = self._extract_project(d["path"], self._root_path)
            try:
                d["keywords"] = json.loads(d["keywords"]) if d["keywords"] else []
            except (json.JSONDecodeError, TypeError):
                d["keywords"] = []
            results.append(d)
        return results

    def get_stale_high_value(self, months_old=3, min_score=50, limit=25) -> list[dict]:
        cutoff = (datetime.now() - timedelta(days=months_old * 30)).isoformat()
        rows = self._conn.execute("""
            SELECT f.id, f.filename, f.path, f.relevance_score, f.modified_at,
                   a.category
            FROM files f
            JOIN analyses a ON f.id = a.file_id
            WHERE f.status = 'analyzed'
              AND f.relevance_score >= ?
              AND f.modified_at IS NOT NULL
              AND f.modified_at < ?
            ORDER BY f.relevance_score DESC
            LIMIT ?
        """, (min_score, cutoff, limit)).fetchall()
        now = datetime.now()
        results = []
        for r in rows:
            d = dict(r)
            d["project"] = self._extract_project(d["path"], self._root_path)
            try:
                mod = datetime.fromisoformat(d["modified_at"][:19])
                d["days_stale"] = (now - mod).days
            except (ValueError, TypeError):
                d["days_stale"] = 0
            results.append(d)
        return results

    def get_orphan_specs(self) -> list[dict]:
        all_files = self._get_analyzed_files_with_analysis()
        projects = defaultdict(lambda: {"categories": set(), "specs": []})
        for f in all_files:
            cat = (f.get("category") or "").strip()
            proj = f["project"]
            projects[proj]["categories"].add(cat)
            if cat == "Feature Spec":
                projects[proj]["specs"].append(f)
        results = []
        for proj, data in sorted(projects.items()):
            if data["specs"] and "Code" not in data["categories"]:
                for spec in data["specs"]:
                    results.append({
                        "project": proj,
                        "id": spec["id"],
                        "filename": spec["filename"],
                        "path": spec["path"],
                        "summary": spec.get("summary", ""),
                    })
        return results

    def get_dead_handoffs(self) -> list[dict]:
        all_files = self._get_analyzed_files_with_analysis()
        # Group by project to find latest modified_at per project
        project_latest = defaultdict(lambda: "")
        for f in all_files:
            mod = f.get("modified_at") or ""
            if mod > project_latest[f["project"]]:
                project_latest[f["project"]] = mod

        handoff_pattern = re.compile(r"(next\s+steps?|TODO|next\s+session)", re.IGNORECASE)
        results = []
        for f in all_files:
            cat = (f.get("category") or "").strip()
            if cat != "Session Handoff":
                continue
            text = f.get("text_sample") or ""
            if not handoff_pattern.search(text):
                continue
            # Check if this is the latest file in its project
            proj_latest = project_latest.get(f["project"], "")
            file_mod = f.get("modified_at") or ""
            if file_mod >= proj_latest:
                # Extract a preview of next steps
                preview = ""
                for line in text.split("\n"):
                    if handoff_pattern.search(line):
                        preview = line.strip()[:200]
                        break
                results.append({
                    "project": f["project"],
                    "id": f["id"],
                    "filename": f["filename"],
                    "path": f["path"],
                    "modified_at": f["modified_at"],
                    "preview": preview,
                })
        return results

    def get_shared_filenames(self, min_projects=3) -> list[dict]:
        all_files = self._get_analyzed_files_with_analysis()
        by_filename = defaultdict(list)
        for f in all_files:
            by_filename[f["filename"]].append(f)
        results = []
        for filename, files in sorted(by_filename.items()):
            projects = {}
            for f in files:
                proj = f["project"]
                if proj not in projects:
                    projects[proj] = f.get("category", "")
            if len(projects) >= min_projects:
                results.append({
                    "filename": filename,
                    "count": len(projects),
                    "projects": projects,  # {project_name: category}
                })
        results.sort(key=lambda x: -x["count"])
        return results

    def get_tech_matrix(self, project_groups: dict) -> dict:
        """Returns {keywords: [str], groups: [str], matrix: {kw: {group: count}}}."""
        all_files = self._get_analyzed_files_with_analysis()
        # Build reverse lookup: project -> group
        proj_to_group = {}
        for group, members in project_groups.items():
            for m in members:
                proj_to_group[m] = group

        # Count keywords per group
        kw_groups = defaultdict(lambda: Counter())
        for f in all_files:
            group = proj_to_group.get(f["project"])
            if not group:
                continue
            for kw in f["keywords"]:
                clean = kw.strip().lower()
                if self._is_valid_keyword(clean):
                    kw_groups[clean][group] += 1

        # Filter to keywords in 2+ groups
        matrix = {}
        for kw, group_counts in kw_groups.items():
            if len(group_counts) >= 2:
                matrix[kw] = dict(group_counts)

        # Get ordered list of groups that appear
        all_groups_set = set()
        for gc in matrix.values():
            all_groups_set.update(gc.keys())
        groups = sorted(all_groups_set)

        # Sort keywords by total count descending
        sorted_kws = sorted(matrix.keys(), key=lambda k: -sum(matrix[k].values()))

        return {"keywords": sorted_kws[:40], "groups": groups, "matrix": matrix}

    def get_category_balance(self, project_groups: dict) -> list[dict]:
        all_files = self._get_analyzed_files_with_analysis()
        proj_to_group = {}
        for group, members in project_groups.items():
            for m in members:
                proj_to_group[m] = group

        group_cats = defaultdict(Counter)
        for f in all_files:
            group = proj_to_group.get(f["project"])
            if not group:
                continue
            cat = (f.get("category") or "Uncategorized").strip()
            group_cats[group][cat] += 1

        results = []
        for group in sorted(group_cats.keys()):
            cats = group_cats[group]
            total = sum(cats.values())
            if total == 0:
                continue
            breakdown = {c: round(n / total * 100) for c, n in cats.most_common()}
            spec_pct = sum(v for k, v in breakdown.items() if "Spec" in k)
            code_pct = breakdown.get("Code", 0)
            flag = None
            if spec_pct > 40:
                flag = "over-specced"
            elif code_pct > 60:
                flag = "under-documented"
            results.append({
                "group": group,
                "total": total,
                "breakdown": breakdown,
                "spec_pct": spec_pct,
                "code_pct": code_pct,
                "flag": flag,
            })
        results.sort(key=lambda x: -x["total"])
        return results

    def get_keyword_clusters(self, min_shared=3, min_cluster_size=3) -> list[dict]:
        all_files = self._get_analyzed_files_with_analysis()
        # Build keyword sets per file
        file_kws = []
        for f in all_files:
            kws = {k.strip().lower() for k in f["keywords"] if self._is_valid_keyword(k.strip().lower())}
            if len(kws) >= min_shared:
                file_kws.append({"file": f, "kws": kws})

        # Greedy clustering via pairwise overlap
        clusters = []
        used = set()
        for i in range(len(file_kws)):
            if i in used:
                continue
            cluster_files = [i]
            cluster_kws = set(file_kws[i]["kws"])
            for j in range(i + 1, len(file_kws)):
                if j in used:
                    continue
                overlap = cluster_kws & file_kws[j]["kws"]
                if len(overlap) >= min_shared:
                    cluster_files.append(j)
                    cluster_kws = cluster_kws & file_kws[j]["kws"]
                    used.add(j)
            if len(cluster_files) >= min_cluster_size:
                used.update(cluster_files)
                members = []
                for idx in cluster_files:
                    f = file_kws[idx]["file"]
                    members.append({
                        "id": f["id"],
                        "filename": f["filename"],
                        "project": f["project"],
                    })
                # Recompute shared keywords across all members
                shared = file_kws[cluster_files[0]]["kws"]
                for idx in cluster_files[1:]:
                    shared = shared & file_kws[idx]["kws"]
                clusters.append({
                    "shared_keywords": sorted(shared),
                    "files": members,
                })

        clusters.sort(key=lambda x: -len(x["files"]))
        return clusters[:15]

    def get_keyword_distribution(self) -> dict:
        all_files = self._get_analyzed_files_with_analysis()
        # keyword -> set of projects
        kw_projects = defaultdict(set)
        for f in all_files:
            for kw in f["keywords"]:
                clean = kw.strip().lower()
                if self._is_valid_keyword(clean):
                    kw_projects[clean].add(f["project"])

        unique = []  # 1 project only
        universal = []  # 5+ projects
        for kw, projs in sorted(kw_projects.items()):
            if len(projs) == 1:
                unique.append({"keyword": kw, "project": next(iter(projs))})
            elif len(projs) >= 5:
                universal.append({"keyword": kw, "project_count": len(projs)})

        unique.sort(key=lambda x: x["keyword"])
        universal.sort(key=lambda x: -x["project_count"])
        return {"unique": unique[:50], "universal": universal}

    def get_documentation_gaps(self) -> list[dict]:
        all_files = self._get_analyzed_files_with_analysis()
        project_cats = defaultdict(set)
        for f in all_files:
            cat = (f.get("category") or "").strip()
            if cat:
                project_cats[f["project"]].add(cat)

        gaps = []
        checks = [
            ("Code", "Testing", "needs tests"),
            ("Code", "Architecture", "needs design docs"),
            ("Code", "README", "needs onboarding docs"),
        ]
        for proj, cats in sorted(project_cats.items()):
            missing = []
            for has_cat, needs_cat, label in checks:
                if has_cat in cats and needs_cat not in cats:
                    missing.append(label)
            # Also check for Setup category
            if "Code" in cats and "Setup" not in cats and "README" not in cats:
                if "needs onboarding docs" not in missing:
                    missing.append("needs onboarding docs")
            if missing:
                gaps.append({"project": proj, "gaps": missing, "categories": sorted(cats)})
        gaps.sort(key=lambda x: -len(x["gaps"]))
        return gaps

    def get_under_analyzed(self, min_sample_length=2000, max_keywords=3, limit=20) -> list[dict]:
        all_files = self._get_analyzed_files_with_analysis()
        results = []
        for f in all_files:
            text_len = len(f.get("text_sample") or "")
            kw_count = len(f["keywords"])
            if text_len >= min_sample_length and kw_count <= max_keywords:
                results.append({
                    "id": f["id"],
                    "filename": f["filename"],
                    "path": f["path"],
                    "project": f["project"],
                    "text_length": text_len,
                    "keyword_count": kw_count,
                    "category": f.get("category", ""),
                    "relevance_score": f.get("relevance_score", 0),
                })
        results.sort(key=lambda x: -x["text_length"])
        return results[:limit]

    def get_insights(self, project_groups: dict) -> dict:
        return {
            "stale": self.get_stale_high_value(),
            "orphan_specs": self.get_orphan_specs(),
            "dead_handoffs": self.get_dead_handoffs(),
            "shared_filenames": self.get_shared_filenames(),
            "tech_matrix": self.get_tech_matrix(project_groups),
            "category_balance": self.get_category_balance(project_groups),
            "keyword_clusters": self.get_keyword_clusters(),
            "keyword_dist": self.get_keyword_distribution(),
            "doc_gaps": self.get_documentation_gaps(),
            "under_analyzed": self.get_under_analyzed(),
        }
