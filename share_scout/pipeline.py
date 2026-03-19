"""Orchestrate the crawl pipeline: scan → score → extract → LLM → catalog."""

import logging
import sys

from .catalog import Catalog
from .checkpoint import CheckpointManager
from .config import load_config, load_scoring_rules, apply_cli_overrides, get_llm_provider
from .extractor import extract_text, compute_partial_hash
from .llm_client import analyze, check_llm
from .scanner import scan_files
from .scorer import Scorer

logger = logging.getLogger(__name__)


def run_crawl(config: dict, rules: dict, dry_run: bool = False):
    """Run the full crawl pipeline in two phases:
    Phase 1: Walk + score + extract all files (fast, no LLM)
    Phase 2: LLM-analyze extracted files in priority order
    """
    crawl_cfg = config["crawl"]
    root_path = crawl_cfg["root_path"]
    batch_size = crawl_cfg["batch_size"]
    max_chars = crawl_cfg["text_sample_max_chars"]
    hash_bytes = crawl_cfg["hash_bytes"]

    scorer = Scorer(rules)

    if dry_run:
        _run_dry(root_path, crawl_cfg, scorer)
        return

    catalog = Catalog(config["catalog"]["db_path"])
    with catalog.connection():
        catalog.init_schema()
        ckpt = CheckpointManager(catalog)
        ckpt.start_run(root_path)

        # ── Phase 1: Walk + Score + Extract (fast) ──
        logger.info("Phase 1: Scanning and extracting all files...")
        batch = []
        total_scanned = 0
        total_new = 0
        total_skipped = 0
        total_extracted = 0

        try:
            for file_meta in scan_files(
                root_path,
                skip_hidden=crawl_cfg["skip_hidden"],
                skip_dirs=crawl_cfg["skip_dirs"],
            ):
                total_scanned += 1

                # Skip files already in DB (any status except pending)
                if ckpt.should_skip(file_meta["path"]):
                    continue

                total_new += 1

                # Score
                score, skip_reason = scorer.score(file_meta)
                file_meta["relevance_score"] = score

                if skip_reason:
                    file_meta["status"] = "skipped"
                    file_meta["skip_reason"] = skip_reason
                    file_meta["content_hash"] = None
                    total_skipped += 1
                    batch.append((file_meta, None, None))
                else:
                    # Extract text + hash (fast, no LLM)
                    text = extract_text(file_meta["path"], max_chars=max_chars)
                    content_hash = compute_partial_hash(file_meta["path"], hash_bytes)
                    file_meta["content_hash"] = content_hash
                    file_meta["status"] = "extracted"
                    total_extracted += 1
                    # Store text for phase 2 but don't analyze yet
                    batch.append((file_meta, text, None))

                # Commit in batches of 100 (fast, no LLM bottleneck)
                if len(batch) >= batch_size:
                    _commit_batch(catalog, batch)
                    ckpt.record_batch(found=len(batch), analyzed=0)
                    batch = []
                    if total_new % 500 == 0:
                        logger.info(
                            "Phase 1: %d scanned, %d new (%d extracted, %d skipped)",
                            total_scanned, total_new, total_extracted, total_skipped,
                        )

            # Commit remaining
            if batch:
                _commit_batch(catalog, batch)
                ckpt.record_batch(found=len(batch), analyzed=0)

            logger.info(
                "Phase 1 complete: %d scanned, %d new (%d extracted, %d skipped)",
                total_scanned, total_new, total_extracted, total_skipped,
            )

            # ── Phase 2: LLM Analysis (slow, priority order) ──
            llm_available = check_llm(config)
            if not llm_available:
                logger.warning(
                    "LLM not reachable — skipping analysis. "
                    "Configure Ollama or an OpenAI-compatible API and re-run.",
                )
                ckpt.complete()
                return

            provider = get_llm_provider(config)
            model_name = config.get(provider, {}).get("model", "unknown") if provider != "none" else "none"
            logger.info(
                "Phase 2: LLM analysis (model: %s). Analyzing extracted files by score...",
                model_name,
            )

            total_analyzed = 0
            while True:
                # Fetch next batch of extracted files, highest score first
                pending = catalog.get_pending_files(limit=10)
                if not pending:
                    break

                for file_row in pending:
                    # Re-extract text (not stored in files table)
                    text = extract_text(file_row["path"], max_chars=max_chars)
                    if not text:
                        catalog.upsert_file({**file_row, "status": "skipped", "skip_reason": "no text extracted"})
                        catalog.commit()
                        continue

                    # Fetch similar files for context
                    similar = catalog.get_similar_files(file_row["filename"])

                    analysis = analyze(
                        file_row, text, config,
                        similar_context=similar if similar else None,
                    )

                    if analysis:
                        catalog.insert_analysis(
                            file_row["id"],
                            text_sample=text,
                            summary=analysis["summary"],
                            keywords=analysis["keywords"],
                            category=analysis["category"],
                            llm_stats=analysis.get("llm_stats"),
                        )
                        catalog.upsert_file({**file_row, "status": "analyzed"})
                        total_analyzed += 1
                    else:
                        # LLM failed — mark as extracted so we retry later
                        catalog.upsert_file({**file_row, "status": "extracted"})

                    catalog.commit()
                    ckpt.record_batch(found=0, analyzed=1 if analysis else 0)

                    if total_analyzed % 10 == 0 and total_analyzed > 0:
                        remaining = catalog._conn.execute(
                            "SELECT COUNT(*) as cnt FROM files WHERE status = 'extracted'"
                        ).fetchone()["cnt"]
                        logger.info(
                            "Phase 2: %d analyzed, %d remaining",
                            total_analyzed, remaining,
                        )

            ckpt.complete()

        except KeyboardInterrupt:
            logger.info("Crawl interrupted — progress saved. Resume by running again.")
            sys.exit(0)

    logger.info("Crawl complete: %d analyzed total", total_analyzed)


def _commit_batch(catalog: Catalog, batch: list):
    """Commit a batch of files and analyses to the catalog."""
    for file_meta, text, analysis in batch:
        catalog.upsert_file(file_meta)

    catalog.commit()

    # Insert analyses after files are committed (need file IDs)
    for file_meta, text, analysis in batch:
        if analysis and text:
            file_row = catalog.get_file_by_path(file_meta["path"])
            if file_row:
                catalog.insert_analysis(
                    file_row["id"],
                    text_sample=text,
                    summary=analysis["summary"],
                    keywords=analysis["keywords"],
                    category=analysis["category"],
                    llm_stats=analysis.get("llm_stats"),
                )
                catalog.upsert_file({**file_meta, "status": "analyzed"})

    catalog.commit()


def _run_dry(root_path: str, crawl_cfg: dict, scorer: Scorer):
    """Dry-run: walk + score only, print summary."""
    brackets = {"0-19": 0, "20-39": 0, "40-59": 0, "60-79": 0, "80-100": 0, "negative": 0}
    ext_counts = {}
    total = 0
    would_skip = 0
    would_analyze = 0

    for file_meta in scan_files(
        root_path,
        skip_hidden=crawl_cfg["skip_hidden"],
        skip_dirs=crawl_cfg["skip_dirs"],
    ):
        total += 1
        score, skip_reason = scorer.score(file_meta)
        ext = file_meta.get("extension", "(none)")
        ext_counts[ext] = ext_counts.get(ext, 0) + 1

        if skip_reason:
            would_skip += 1
        else:
            would_analyze += 1

        if score < 0:
            brackets["negative"] += 1
        elif score < 20:
            brackets["0-19"] += 1
        elif score < 40:
            brackets["20-39"] += 1
        elif score < 60:
            brackets["40-59"] += 1
        elif score < 80:
            brackets["60-79"] += 1
        else:
            brackets["80-100"] += 1

    print(f"\n{'='*50}")
    print(f"DRY RUN SUMMARY: {root_path}")
    print(f"{'='*50}")
    print(f"Total files found: {total}")
    print(f"Would analyze:     {would_analyze}")
    print(f"Would skip:        {would_skip}")
    print(f"\nScore distribution:")
    for bracket, count in brackets.items():
        bar = "#" * min(count, 50)
        print(f"  {bracket:>8}: {count:>6}  {bar}")
    print(f"\nTop extensions:")
    for ext, count in sorted(ext_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {(ext or '(none)'):>8}: {count}")
    print()
