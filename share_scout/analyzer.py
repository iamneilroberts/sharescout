"""Per-file analysis orchestrator — decides strategy, manages chunks, handles image captioning."""

import logging
from collections import Counter

from .catalog import Catalog
from .extractor import extract_structured, DocumentContent, ExtractedImage
from .llm_client import analyze, caption_image, detect_context_budget
from .prompts import get_prompt, get_categories, get_summary_length

logger = logging.getLogger(__name__)

# Log vision unavailability only once
_vision_warned = False


def analyze_file(
    file_row: dict,
    config: dict,
    catalog: Catalog,
    context_budget_chars: int,
    seen_image_hashes: Counter,
    verbose: bool = False,
) -> dict | None:
    """Analyze a single file using adaptive strategy based on content size.

    Returns dict with keys: summary, keywords, category, llm_stats,
    processing_strategy, chunk_count, image_captions, total_chars_extracted,
    context_budget_used.
    Returns None if analysis fails entirely.
    """
    preset = config.get("_preset", {})

    # Extract structured content
    doc = extract_structured(file_row["path"], preset_config=preset)
    if not doc or not doc.sections:
        if verbose:
            logger.debug("No structured content extracted for %s", file_row["path"])
        return None

    # Determine strategy
    strategy = _select_strategy(doc.total_chars, context_budget_chars, preset)

    if verbose:
        logger.info(
            "Analyzing %s — strategy=%s, total_chars=%d, budget=%d, sections=%d, images=%d",
            file_row["filename"], strategy, doc.total_chars, context_budget_chars,
            len(doc.sections), len(doc.images),
        )

    # Process images (shared across strategies)
    image_captions, skipped_images = _process_images(
        doc.images, config, seen_image_hashes, verbose,
    )

    # Get prompt configuration from preset
    prompt_template = get_prompt("analysis", preset)
    categories = get_categories(preset)
    summary_length = get_summary_length(strategy, preset)

    # Fetch similar files for context
    similar = catalog.get_similar_files(file_row["filename"])

    if strategy == "single":
        result = _analyze_single(
            file_row, doc, image_captions, config, similar,
            prompt_template, categories, summary_length, verbose,
        )
    elif strategy == "sampled":
        result = _analyze_sampled(
            file_row, doc, image_captions, config, similar,
            context_budget_chars, prompt_template, categories, summary_length, verbose,
        )
    else:  # chunked
        result = _analyze_chunked(
            file_row, doc, image_captions, config, catalog, similar,
            context_budget_chars, preset, prompt_template, categories, verbose,
        )

    if result:
        result["processing_strategy"] = strategy
        result["total_chars_extracted"] = doc.total_chars
        result["context_budget_used"] = context_budget_chars
        result["image_captions"] = [cap for _, cap in image_captions] if image_captions else []
        result["images_found"] = len(doc.images)
        result["images_captioned"] = len(image_captions)
        result["images_skipped_no_vision"] = skipped_images.get("no_vision", 0)
        result["images_skipped_dedup"] = skipped_images.get("dedup", 0)
        result["images_skipped_failed"] = skipped_images.get("failed", 0)
        if strategy != "chunked":
            result["chunk_count"] = None

    return result


def _select_strategy(total_chars: int, budget: int, preset: dict) -> str:
    """Select processing strategy based on content size vs context budget."""
    if total_chars <= budget * 0.8:
        return "single"
    elif total_chars <= budget * 2:
        return "sampled"
    else:
        return "chunked"


def _analyze_single(
    file_row, doc, image_captions, config, similar,
    prompt_template, categories, summary_length, verbose,
):
    """Single-pass: send all text to LLM."""
    text = _assemble_text(doc.sections, image_captions)

    if verbose:
        logger.debug("Single-pass text length: %d chars", len(text))

    result = analyze(
        file_row, text, config,
        similar_context=similar if similar else None,
        prompt_template=prompt_template,
        categories=categories,
        summary_length=summary_length,
    )
    if result:
        result["_text_sample"] = text[:8000]  # store a reasonable sample
    return result


def _analyze_sampled(
    file_row, doc, image_captions, config, similar,
    budget, prompt_template, categories, summary_length, verbose,
):
    """Multi-section sample: select representative sections within budget."""
    selected = _select_sample_sections(doc.sections, budget)

    text = _assemble_text(selected, image_captions)

    if verbose:
        logger.debug(
            "Sampled %d of %d sections, text length: %d chars",
            len(selected), len(doc.sections), len(text),
        )

    result = analyze(
        file_row, text, config,
        similar_context=similar if similar else None,
        prompt_template=prompt_template,
        categories=categories,
        summary_length=summary_length,
    )
    if result:
        result["_text_sample"] = text[:8000]
    return result


def _analyze_chunked(
    file_row, doc, image_captions, config, catalog, similar,
    budget, preset, prompt_template, categories, verbose,
):
    """Chunked: split into chunks, summarize each, then rollup."""
    max_chunks = preset.get("analysis", {}).get("max_chunks", 10)
    chunks = _build_chunks(doc.sections, budget, max_chunks)

    if verbose:
        logger.info("Chunked analysis: %d chunks for %s", len(chunks), file_row["filename"])

    # Check for already-processed chunks (resume support)
    file_id = file_row.get("id")
    completed_indices = set()
    existing_summaries = {}
    if file_id:
        completed_indices = catalog.get_completed_chunk_indices(file_id)
        if completed_indices:
            for cs in catalog.get_chunk_summaries(file_id):
                existing_summaries[cs["chunk_index"]] = cs["chunk_summary"]

    chunk_summary_texts = []
    chunk_analysis_cats = get_categories(preset)

    for i, chunk_sections in enumerate(chunks):
        if i in completed_indices and i in existing_summaries:
            if verbose:
                logger.debug("  Chunk %d: already processed (resume)", i)
            chunk_summary_texts.append(existing_summaries[i])
            continue

        chunk_text = _assemble_text(chunk_sections, image_captions)

        if verbose:
            logger.debug("  Chunk %d: %d chars", i, len(chunk_text))

        # Use a chunk-specific prompt
        chunk_prompt = (
            f"Summarize this section (chunk {i+1} of {len(chunks)}) of a larger document. "
            f"Extract the key information, specific details, and important names/terms.\n\n"
            f"File: {file_row.get('filename', '')}\n"
            f"Path: {file_row.get('path', '')}\n\n"
            f"Text:\n---\n{chunk_text}\n---\n\n"
            f"Respond in JSON: {{\"summary\": \"...\", \"keywords\": [\"...\"], \"category\": \"...\"}}\n"
            f"Category (pick one): {chunk_analysis_cats}"
        )

        chunk_result = analyze(
            file_row, chunk_text, config,
            prompt_template=chunk_prompt,
            categories=chunk_analysis_cats,
            summary_length="3-5 sentences",
        )

        if chunk_result:
            chunk_summary = chunk_result["summary"]
            chunk_summary_texts.append(chunk_summary)

            if verbose:
                logger.debug("  Chunk %d summary: %s", i, chunk_summary[:200])

            # Store chunk summary in DB
            if file_id:
                catalog.insert_chunk_summary(
                    file_id, i, chunk_text[:4000], chunk_summary,
                    llm_stats=chunk_result.get("llm_stats"),
                )
                catalog.commit()
        else:
            chunk_summary_texts.append(f"[Chunk {i+1}: analysis failed]")

    if not any(s for s in chunk_summary_texts if not s.startswith("[")):
        return None

    # Rollup: synthesize chunk summaries
    rollup_template = get_prompt("rollup", preset)
    summary_length = get_summary_length("chunked", preset)

    rollup_input = "\n\n".join(
        f"--- Section {i+1} ---\n{s}" for i, s in enumerate(chunk_summary_texts)
    )

    if verbose:
        logger.info("Rollup input (%d chunk summaries, %d chars)", len(chunk_summary_texts), len(rollup_input))

    rollup_prompt = rollup_template.format(
        filename=file_row.get("filename", ""),
        path=file_row.get("path", ""),
        size_bytes=file_row.get("size_bytes", 0),
        chunk_count=len(chunk_summary_texts),
        chunk_summaries=rollup_input,
        summary_length=summary_length,
        categories=categories,
    )

    result = analyze(
        file_row, rollup_input, config,
        similar_context=similar if similar else None,
        prompt_template=rollup_prompt,
        categories=categories,
        summary_length=summary_length,
    )

    if result:
        result["chunk_count"] = len(chunks)
        result["_text_sample"] = rollup_input[:8000]

        if verbose:
            logger.info("Rollup result: %s", result["summary"][:200])

    return result


def _process_images(
    images: list[ExtractedImage],
    config: dict,
    seen_image_hashes: Counter,
    verbose: bool,
) -> tuple[list[tuple[int, str]], dict]:
    """Process images: dedup and caption.

    Returns (captions, skipped) where:
        captions: list of (char_offset, caption) tuples
        skipped: dict with counts {no_vision, dedup, failed}
    """
    global _vision_warned

    skipped = {"no_vision": 0, "dedup": 0, "failed": 0}

    if not images:
        return [], skipped

    preset = config.get("_preset", {})
    vision_model = preset.get("llm", {}).get("vision_model")
    if not vision_model:
        vision_model = config.get("llm", {}).get("vision_model")

    if not vision_model:
        skipped["no_vision"] = len(images)
        if not _vision_warned:
            logger.warning(
                "No vision model configured — %d image(s) in this document will not be captioned. "
                "Set llm.vision_model in your preset or config to enable image processing.",
                len(images),
            )
            _vision_warned = True
        elif verbose:
            logger.debug("Skipping %d images (no vision model)", len(images))
        return [], skipped

    dedup_threshold = preset.get("image_dedup_threshold", 3)
    caption_prompt = get_prompt("image_caption", preset)

    vision_config = {
        "model": vision_model,
        "provider": preset.get("llm", {}).get("vision_provider", "ollama"),
    }

    captions = []
    for img in images:
        # Dedup check
        count = seen_image_hashes[img.content_hash]
        if count >= dedup_threshold:
            skipped["dedup"] += 1
            if verbose:
                logger.debug("Skipping deduped image %s (seen %d times)", img.source, count)
            continue

        seen_image_hashes[img.content_hash] += 1

        caption = caption_image(img.data, caption_prompt, config, vision_config)
        if caption:
            captions.append((img.char_offset, caption))
            if verbose:
                logger.debug("Image caption (%s): %s", img.source, caption[:100])
        else:
            skipped["failed"] += 1

    return captions, skipped


def _assemble_text(sections, image_captions):
    """Concatenate sections with inline image captions at appropriate positions."""
    if not image_captions:
        return "\n\n".join(s.text for s in sections)

    # Sort captions by offset
    sorted_captions = sorted(image_captions, key=lambda x: x[0])

    parts = []
    caption_idx = 0
    for section in sections:
        # Insert any captions that belong before/within this section
        while caption_idx < len(sorted_captions):
            cap_offset, cap_text = sorted_captions[caption_idx]
            if cap_offset <= section.char_offset + len(section.text):
                parts.append(f"[Image: {cap_text}]")
                caption_idx += 1
            else:
                break
        parts.append(section.text)

    # Append remaining captions
    while caption_idx < len(sorted_captions):
        _, cap_text = sorted_captions[caption_idx]
        parts.append(f"[Image: {cap_text}]")
        caption_idx += 1

    return "\n\n".join(parts)


def _select_sample_sections(sections, budget):
    """Select representative sections: first 30%, heading sections 40%, last 30%."""
    if not sections:
        return []

    total_chars = sum(len(s.text) for s in sections)
    if total_chars <= budget:
        return sections

    target = int(budget * 0.9)  # leave some room
    selected = []
    selected_chars = 0

    # Categorize sections
    heading_sections = [s for s in sections if s.position.startswith("heading:")]
    non_heading = [s for s in sections if not s.position.startswith("heading:")]

    # First ~30% of budget from the beginning
    begin_budget = int(target * 0.3)
    for s in sections:
        if selected_chars >= begin_budget:
            break
        selected.append(s)
        selected_chars += len(s.text)

    # Heading sections for ~40% of budget
    heading_budget = int(target * 0.4)
    heading_chars = 0
    for s in heading_sections:
        if s in selected:
            continue
        if heading_chars >= heading_budget:
            break
        selected.append(s)
        heading_chars += len(s.text)
        selected_chars += len(s.text)

    # Last ~30% from the end
    end_budget = target - selected_chars
    end_sections = []
    end_chars = 0
    for s in reversed(sections):
        if s in selected:
            continue
        if end_chars >= end_budget:
            break
        end_sections.insert(0, s)
        end_chars += len(s.text)

    selected.extend(end_sections)

    # Sort by original offset
    selected.sort(key=lambda s: s.char_offset)
    return selected


def _build_chunks(sections, budget, max_chunks):
    """Group sections into chunks that each fit within the context budget."""
    chunks = []
    current_chunk = []
    current_size = 0
    chunk_budget = int(budget * 0.85)  # leave room for prompt overhead

    for section in sections:
        section_size = len(section.text)

        # If a single section exceeds budget, split it
        if section_size > chunk_budget:
            # Flush current chunk
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_size = 0

            # Split the large section into sub-sections
            text = section.text
            offset = section.char_offset
            while text:
                part = text[:chunk_budget]
                text = text[chunk_budget:]
                from .extractor import TextSection
                chunks.append([TextSection(text=part, position=section.position, char_offset=offset)])
                offset += len(part)
            continue

        if current_size + section_size > chunk_budget:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = [section]
            current_size = section_size
        else:
            current_chunk.append(section)
            current_size += section_size

    if current_chunk:
        chunks.append(current_chunk)

    # Limit to max_chunks by merging smaller chunks
    while len(chunks) > max_chunks and len(chunks) > 1:
        # Find smallest adjacent pair to merge
        min_size = float("inf")
        min_idx = 0
        for i in range(len(chunks) - 1):
            combined = sum(len(s.text) for s in chunks[i]) + sum(len(s.text) for s in chunks[i+1])
            if combined < min_size:
                min_size = combined
                min_idx = i
        chunks[min_idx] = chunks[min_idx] + chunks[min_idx + 1]
        chunks.pop(min_idx + 1)

    return chunks
