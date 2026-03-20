"""Prompt templates for LLM analysis, parameterized by domain preset."""

DEFAULT_ANALYSIS_PROMPT = """You are building a knowledge base catalog from a developer's projects. Your job: help the developer find specific features, decisions, and implementations across projects by writing summaries that surface WHAT MAKES EACH FILE UNIQUE.

Analyze the text excerpt thoroughly. Extract:
- Specific feature names, API endpoints, database tables, or UI components mentioned
- Technology choices and architectural decisions (what stack, what patterns, why)
- Status: is this planned, in-progress, completed, or abandoned?
- Key differences from similar files (see context below)

Rules for the summary:
- {summary_length}. Never start with "This document/file/README". Start with the subject.
- Be SPECIFIC: mention actual feature names, tech, endpoints, table names — not generic descriptions
- If context shows similar files exist, explicitly state what's DIFFERENT about this one
- Good: "Booking flow implementation using Xata DB with multi-step wizard and Resend email confirmation. Differs from claude-travel-agent-v2 version by adding approval workflows and cancellation support."
- Bad: "Feature specification for a booking system in the travel assistant project."

{context_block}
File: {filename}
Path: {path}
Extension: {extension}
Size: {size_bytes} bytes

Text excerpt:
---
{text_sample}
---

Respond in this exact JSON format (no other text):
{{"summary": "...", "keywords": ["...", "..."], "category": "..."}}

Keywords: 3-7 specific terms. Include: exact feature names, technologies, libraries, the project name.

Category (pick one): {categories}"""

DEFAULT_ROLLUP_PROMPT = """You are synthesizing section-by-section summaries of a large document into one cohesive overview for a knowledge base catalog.

File: {filename}
Path: {path}
Total size: {size_bytes} bytes
Sections summarized: {chunk_count}

Section summaries:
---
{chunk_summaries}
---

Write a {summary_length} summary that:
1. Captures the document's overall purpose and scope
2. Highlights the most important specific details (names, technologies, decisions)
3. Notes any structural patterns (e.g., "covers 5 policy areas" or "defines 12 API endpoints")
4. Never starts with "This document"

Also provide keywords and a category.

Respond in this exact JSON format (no other text):
{{"summary": "...", "keywords": ["...", "..."], "category": "..."}}

Keywords: 3-7 specific terms.

Category (pick one): {categories}"""

DEFAULT_IMAGE_CAPTION_PROMPT = """Describe this image concisely for document cataloging purposes. Focus on:
- What type of image it is (diagram, screenshot, chart, photo, logo, etc.)
- Key information it conveys (labels, data points, relationships)
- How it relates to document content

Keep the description to 1-2 sentences. Be specific about any text, numbers, or labels visible."""

DEFAULT_CATEGORIES = '"Feature Spec", "Architecture", "Session Handoff", "README/Setup", "Configuration", "API/Integration", "Database", "Testing", "Deployment", "User Guide", "Development Journal", "Code", "Data", "Other"'

SUMMARY_LENGTHS = {
    "short": "2-3 sentences",
    "medium": "5-10 sentences",
    "long": "15-30 sentences",
}


def get_prompt(name: str, preset_config: dict) -> str:
    """Return the prompt template for the given name, using preset override if available.

    Args:
        name: One of "analysis", "rollup", "image_caption"
        preset_config: The loaded preset dict (may be empty)

    Returns:
        Prompt template string with {placeholders} for formatting.
    """
    preset_prompts = preset_config.get("prompts", {})

    if name in preset_prompts and preset_prompts[name]:
        return preset_prompts[name]

    defaults = {
        "analysis": DEFAULT_ANALYSIS_PROMPT,
        "rollup": DEFAULT_ROLLUP_PROMPT,
        "image_caption": DEFAULT_IMAGE_CAPTION_PROMPT,
    }
    return defaults.get(name, "")


def get_categories(preset_config: dict) -> str:
    """Return comma-separated category list from preset or default."""
    cats = preset_config.get("categories")
    if cats and isinstance(cats, list):
        return ", ".join(f'"{c}"' for c in cats)
    return DEFAULT_CATEGORIES


def get_summary_length(strategy: str, preset_config: dict) -> str:
    """Return summary length guidance based on strategy and preset config.

    Args:
        strategy: "single", "sampled", or "chunked"
        preset_config: The loaded preset dict
    """
    analysis_cfg = preset_config.get("analysis", {})

    if strategy == "single":
        return analysis_cfg.get("summary_sentences_short", SUMMARY_LENGTHS["short"])
    elif strategy == "sampled":
        return analysis_cfg.get("summary_sentences_medium", SUMMARY_LENGTHS["medium"])
    else:
        return analysis_cfg.get("summary_sentences_long", SUMMARY_LENGTHS["long"])
