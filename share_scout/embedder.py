"""Embedder: generate and store vector embeddings for analyzed catalog files.

Embedding strategy by text size:
  - Full (≤ CHUNK_SIZE chars):    Single embedding of full text
  - Chunked (≤ SAMPLE_THRESHOLD): Split into overlapping chunks, embed each
  - Sampled (> SAMPLE_THRESHOLD):  Select representative sections, then chunk those
"""

import logging

from .catalog import Catalog
from .llm_client import check_embedding_model, generate_embedding

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 50

# nomic-embed-text has 8192 token context. Code/markdown tokenizes at ~1.2 chars/token,
# so 4000 chars is safe even for dense code. Plain prose could go higher but this covers all content.
CHUNK_SIZE = 4000

# Overlap between chunks to preserve context at boundaries
CHUNK_OVERLAP = 400

# Above this, use sampled strategy instead of embedding everything
SAMPLE_THRESHOLD = 100000

# Budget for sampled strategy — how much text to select from very large files
SAMPLE_BUDGET = 40000


def run_embed(config: dict) -> None:
    """Generate embeddings for all analyzed catalog files that don't yet have them."""
    if not check_embedding_model(config):
        embedding_model = config.get("ollama", {}).get("embedding_model")
        if embedding_model:
            logger.error(
                "Embedding model '%s' is not available — check your Ollama endpoint",
                embedding_model,
            )
        else:
            logger.error(
                "No embedding_model configured — set ollama.embedding_model in config.yaml"
            )
        return

    embedding_model = config["ollama"]["embedding_model"]
    batch_size = config.get("crawl", {}).get("batch_size", DEFAULT_BATCH_SIZE)
    db_path = config.get("catalog", {}).get("db_path", "share_scout.db")

    catalog = Catalog(db_path)
    with catalog.connection():
        catalog.init_schema()
        unembedded = catalog.get_unembedded_files()
        total = len(unembedded)

        if total == 0:
            logger.info("No files require embedding — catalog is up to date")
            return

        logger.info("Starting embedding run: %d files to process (model: %s)", total, embedding_model)

        embedded_count = 0
        skipped_count = 0
        total_vectors = 0
        batch_count = 0

        for file_record in unembedded:
            file_id = file_record["id"]
            filename = file_record.get("filename", "<unknown>")

            # Check for chunk summaries first (already chunked during analysis)
            chunks = catalog.get_chunk_summaries(file_id)

            if chunks:
                vectors = _embed_pre_chunked(config, catalog, file_id, filename, chunks, embedding_model)
            else:
                text_sample = file_record.get("text_sample", "")
                if not text_sample:
                    logger.debug("Skipping file %s — no text_sample available", filename)
                    skipped_count += 1
                    continue
                vectors = _embed_text(config, catalog, file_id, filename, text_sample, embedding_model)

            if vectors > 0:
                embedded_count += 1
                total_vectors += vectors
            else:
                skipped_count += 1

            batch_count += 1
            if batch_count >= batch_size:
                catalog.commit()
                batch_count = 0
                logger.info(
                    "Embedded %d/%d files (%d vectors, %d skipped)...",
                    embedded_count, total, total_vectors, skipped_count,
                )

        # Final commit
        if batch_count > 0:
            catalog.commit()

        logger.info(
            "Embedding complete: %d/%d files embedded (%d vectors, %d skipped)",
            embedded_count, total, total_vectors, skipped_count,
        )


def _embed_pre_chunked(config, catalog, file_id, filename, chunks, model):
    """Embed files that already have chunk_summaries from analysis."""
    vectors = 0
    for chunk in chunks:
        chunk_text = chunk.get("chunk_text", "")
        if not chunk_text:
            continue

        # Chunk text from analysis may still exceed embedding context
        text_pieces = _chunk_text(chunk_text) if len(chunk_text) > CHUNK_SIZE else [chunk_text]

        for i, piece in enumerate(text_pieces):
            vector = generate_embedding(config, piece)
            if vector is None:
                logger.warning("Failed to embed chunk %d of %s — skipping file", chunk["chunk_index"], filename)
                return 0
            # Use sub-index for pieces of a single analysis chunk
            chunk_idx = chunk["chunk_index"] * 100 + i if len(text_pieces) > 1 else chunk["chunk_index"]
            catalog.insert_embedding(
                file_id=file_id,
                chunk_index=chunk_idx,
                source="chunk_text",
                vector=vector,
                model=model,
            )
            vectors += 1

    return vectors


def _embed_text(config, catalog, file_id, filename, text, model):
    """Embed a text_sample, using the appropriate strategy based on size."""
    text_len = len(text)

    if text_len <= CHUNK_SIZE:
        # Full: single embedding
        return _embed_single(config, catalog, file_id, text, model)

    if text_len <= SAMPLE_THRESHOLD:
        # Chunked: split into overlapping chunks, embed each
        strategy = "chunked"
        pieces = _chunk_text(text)
    else:
        # Sampled: select representative sections first, then chunk
        strategy = "sampled"
        sampled = _sample_text(text, SAMPLE_BUDGET)
        pieces = _chunk_text(sampled) if len(sampled) > CHUNK_SIZE else [sampled]

    logger.debug(
        "%s: %s strategy — %d chars → %d chunks",
        filename, strategy, text_len, len(pieces),
    )

    vectors = 0
    for i, piece in enumerate(pieces):
        vector = generate_embedding(config, piece)
        if vector is None:
            logger.warning("Failed to embed chunk %d of %s — skipping file", i, filename)
            return 0
        catalog.insert_embedding(
            file_id=file_id,
            chunk_index=i,
            source="text_sample",
            vector=vector,
            model=model,
        )
        vectors += 1

    return vectors


def _embed_single(config, catalog, file_id, text, model):
    """Embed a single text that fits within the context window."""
    vector = generate_embedding(config, text)
    if vector is None:
        return 0
    catalog.insert_embedding(
        file_id=file_id,
        chunk_index=None,
        source="text_sample",
        vector=vector,
        model=model,
    )
    return 1


def _chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks that fit the embedding model context.

    Tries to break at paragraph boundaries (\n\n) for cleaner chunks.
    Falls back to line boundaries, then hard split.
    """
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + CHUNK_SIZE

        if end >= text_len:
            chunks.append(text[start:])
            break

        # Try to break at a paragraph boundary
        break_point = text.rfind("\n\n", start + CHUNK_SIZE // 2, end)
        if break_point == -1:
            # Try line boundary
            break_point = text.rfind("\n", start + CHUNK_SIZE // 2, end)
        if break_point == -1:
            # Hard split
            break_point = end

        chunks.append(text[start:break_point])
        start = break_point - CHUNK_OVERLAP
        if start < 0:
            start = 0

    return chunks


def _sample_text(text: str, budget: int) -> str:
    """Select representative sections from a very large text.

    Strategy: first 30%, heading/structural lines 40%, last 30%.
    This mirrors the analyzer's _select_sample_sections approach
    but works on raw text instead of structured sections.
    """
    lines = text.split("\n")
    total_chars = len(text)

    # Budget allocation
    begin_budget = int(budget * 0.30)
    heading_budget = int(budget * 0.40)
    end_budget = int(budget * 0.30)

    # First 30%: take lines from the beginning
    begin_lines = []
    begin_chars = 0
    for line in lines:
        if begin_chars >= begin_budget:
            break
        begin_lines.append(line)
        begin_chars += len(line) + 1

    # Heading/structural lines from the middle: lines that look like headings,
    # list items, or key-value pairs — these carry the most semantic signal
    begin_set = set(range(len(begin_lines)))
    heading_lines = []
    heading_chars = 0
    for i, line in enumerate(lines):
        if i in begin_set:
            continue
        stripped = line.strip()
        if not stripped:
            continue
        is_structural = (
            stripped.startswith("#")           # markdown heading
            or stripped.startswith("- ")       # list item
            or stripped.startswith("* ")       # list item
            or stripped.startswith("```")      # code fence
            or ": " in stripped[:60]           # key: value
            or stripped.startswith("def ")     # function def
            or stripped.startswith("class ")   # class def
            or stripped.isupper()              # ALL CAPS heading
        )
        if is_structural:
            if heading_chars >= heading_budget:
                break
            heading_lines.append(line)
            heading_chars += len(line) + 1

    # Last 30%: take lines from the end
    end_lines = []
    end_chars = 0
    used_indices = begin_set | {i for i, l in enumerate(lines) if l in heading_lines}
    for line in reversed(lines):
        if end_chars >= end_budget:
            break
        end_lines.insert(0, line)
        end_chars += len(line) + 1

    # Combine with section markers
    parts = []
    if begin_lines:
        parts.append("\n".join(begin_lines))
    if heading_lines:
        parts.append("[...structural content...]\n" + "\n".join(heading_lines))
    if end_lines:
        parts.append("[...end of document...]\n" + "\n".join(end_lines))

    return "\n\n".join(parts)
