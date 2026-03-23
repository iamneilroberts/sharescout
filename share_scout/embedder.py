"""Embedder: generate and store vector embeddings for analyzed catalog files."""

import logging

from .catalog import Catalog
from .llm_client import check_embedding_model, generate_embedding

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 50


def run_embed(config: dict) -> None:
    """Generate embeddings for all analyzed catalog files that don't yet have them.

    For chunked files (files with chunk_summaries), embeds each chunk's chunk_text.
    For non-chunked files, embeds the text_sample from the analyses table.

    Commits every batch_size files and logs progress.
    """
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
        batch_count = 0

        for file_record in unembedded:
            file_id = file_record["id"]
            filename = file_record.get("filename", "<unknown>")

            # Check for chunk summaries first
            chunks = catalog.get_chunk_summaries(file_id)

            if chunks:
                # Chunked file — embed each chunk's text
                file_ok = True
                for chunk in chunks:
                    chunk_text = chunk.get("chunk_text", "")
                    if not chunk_text:
                        logger.debug("Skipping empty chunk %d for file %s", chunk["chunk_index"], filename)
                        continue
                    vector = generate_embedding(config, chunk_text)
                    if vector is None:
                        logger.warning("Failed to embed chunk %d of %s — skipping file", chunk["chunk_index"], filename)
                        file_ok = False
                        break
                    catalog.insert_embedding(
                        file_id=file_id,
                        chunk_index=chunk["chunk_index"],
                        source="chunk_text",
                        vector=vector,
                        model=embedding_model,
                    )
                if not file_ok:
                    continue
            else:
                # Non-chunked file — embed text_sample
                text_sample = file_record.get("text_sample", "")
                if not text_sample:
                    logger.debug("Skipping file %s — no text_sample available", filename)
                    continue
                vector = generate_embedding(config, text_sample)
                if vector is None:
                    logger.warning("Failed to embed %s — skipping", filename)
                    continue
                catalog.insert_embedding(
                    file_id=file_id,
                    chunk_index=None,
                    source="text_sample",
                    vector=vector,
                    model=embedding_model,
                )

            embedded_count += 1
            batch_count += 1

            if batch_count >= batch_size:
                catalog.commit()
                batch_count = 0
                logger.info("Embedded %d/%d files...", embedded_count, total)

        # Final commit for any remaining files
        if batch_count > 0:
            catalog.commit()

        logger.info("Embedding complete: %d/%d files embedded", embedded_count, total)
