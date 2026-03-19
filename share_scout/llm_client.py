"""LLM client for document summarization and categorization."""

import json
import logging

import httpx
import ollama

from .config import get_llm_provider

logger = logging.getLogger(__name__)

ANALYSIS_PROMPT = """You are building a knowledge base catalog from a developer's projects. Your job: help the developer find specific features, decisions, and implementations across projects by writing summaries that surface WHAT MAKES EACH FILE UNIQUE.

Analyze the text excerpt thoroughly. Extract:
- Specific feature names, API endpoints, database tables, or UI components mentioned
- Technology choices and architectural decisions (what stack, what patterns, why)
- Status: is this planned, in-progress, completed, or abandoned?
- Key differences from similar files (see context below)

Rules for the summary:
- 2-3 sentences. Never start with "This document/file/README". Start with the subject.
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

Category (pick one): "Feature Spec", "Architecture", "Session Handoff", "README/Setup", "Configuration", "API/Integration", "Database", "Testing", "Deployment", "User Guide", "Development Journal", "Code", "Data", "Other"."""


def analyze_document(file_meta: dict, text_sample: str,
                     endpoint: str = "http://localhost:11434",
                     model: str = "llama3.2",
                     timeout: int = 120,
                     similar_context: list[dict] = None) -> dict | None:
    """Send text sample + metadata to Ollama for analysis.

    Returns dict with keys: summary, keywords, category, llm_stats.
    Returns None if analysis fails.
    """
    # Build context block from similar files
    context_block = ""
    if similar_context:
        lines = ["Files with the same or similar name already cataloged (highlight differences from these):"]
        for ctx in similar_context[:5]:
            lines.append(f"  - [{ctx.get('project', '?')}] {ctx['path']}: {ctx['summary']}")
        context_block = "\n".join(lines) + "\n"

    prompt = ANALYSIS_PROMPT.format(
        filename=file_meta.get("filename", ""),
        path=file_meta.get("path", ""),
        extension=file_meta.get("extension", ""),
        size_bytes=file_meta.get("size_bytes", 0),
        text_sample=text_sample[:4000],
        context_block=context_block,
    )

    try:
        client = ollama.Client(host=endpoint, timeout=timeout)
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )

        content = response["message"]["content"].strip()

        # Try to extract JSON from the response
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        result = json.loads(content)

        if not all(k in result for k in ("summary", "keywords", "category")):
            logger.warning("LLM response missing expected keys: %s", content[:200])
            return None

        if isinstance(result["keywords"], str):
            result["keywords"] = [k.strip() for k in result["keywords"].split(",")]

        # Attach Ollama performance stats
        result["llm_stats"] = {
            "total_duration_ms": round(response.get("total_duration", 0) / 1e6),
            "eval_duration_ms": round(response.get("eval_duration", 0) / 1e6),
            "prompt_eval_duration_ms": round(response.get("prompt_eval_duration", 0) / 1e6),
            "eval_count": response.get("eval_count", 0),
            "prompt_eval_count": response.get("prompt_eval_count", 0),
            "load_duration_ms": round(response.get("load_duration", 0) / 1e6),
        }

        return result

    except json.JSONDecodeError as e:
        logger.warning("Failed to parse LLM JSON response: %s", e)
        return None
    except Exception as e:
        logger.warning("LLM analysis failed: %s", e)
        return None


def check_ollama(endpoint: str = "http://localhost:11434") -> bool:
    """Check if Ollama is reachable."""
    try:
        client = ollama.Client(host=endpoint, timeout=5)
        client.list()
        return True
    except Exception:
        return False


def analyze_document_openai(file_meta: dict, text_sample: str,
                             base_url: str, model: str,
                             api_key_env: str = "OPENAI_API_KEY",
                             timeout: int = 60,
                             similar_context: list[dict] = None) -> dict | None:
    """Send text sample to an OpenAI-compatible API for analysis."""
    import os
    api_key = os.environ.get(api_key_env)
    if not api_key:
        logger.warning("API key env var %s not set", api_key_env)
        return None

    context_block = ""
    if similar_context:
        lines = ["Files with the same or similar name already cataloged (highlight differences from these):"]
        for ctx in similar_context[:5]:
            lines.append(f"  - [{ctx.get('project', '?')}] {ctx['path']}: {ctx['summary']}")
        context_block = "\n".join(lines) + "\n"

    prompt = ANALYSIS_PROMPT.format(
        filename=file_meta.get("filename", ""),
        path=file_meta.get("path", ""),
        extension=file_meta.get("extension", ""),
        size_bytes=file_meta.get("size_bytes", 0),
        text_sample=text_sample[:4000],
        context_block=context_block,
    )

    try:
        response = httpx.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}]},
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()

        content = data["choices"][0]["message"]["content"].strip()

        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        result = json.loads(content)

        if not all(k in result for k in ("summary", "keywords", "category")):
            logger.warning("LLM response missing expected keys: %s", content[:200])
            return None

        if isinstance(result["keywords"], str):
            result["keywords"] = [k.strip() for k in result["keywords"].split(",")]

        usage = data.get("usage", {})
        result["llm_stats"] = {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }

        return result

    except httpx.HTTPStatusError as e:
        logger.warning("OpenAI API error %d: %s", e.response.status_code, e.response.text[:200])
        return None
    except httpx.TimeoutException:
        logger.warning("OpenAI API timeout after %ds", timeout)
        return None
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse LLM JSON response: %s", e)
        return None
    except Exception as e:
        logger.warning("LLM analysis failed: %s", e)
        return None


def check_openai(openai_config: dict) -> bool:
    """Check if an OpenAI-compatible API is reachable."""
    import os
    api_key = os.environ.get(openai_config.get("api_key_env", "OPENAI_API_KEY"))
    if not api_key:
        return False
    try:
        base_url = openai_config["base_url"].rstrip("/")
        resp = httpx.get(f"{base_url}/models", headers={"Authorization": f"Bearer {api_key}"}, timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def check_llm(config: dict) -> bool:
    """Check if the configured LLM provider is reachable."""
    provider = get_llm_provider(config)
    if provider == "ollama":
        return check_ollama(config["ollama"]["endpoint"])
    elif provider == "openai":
        return check_openai(config["openai"])
    return False


def analyze(file_meta: dict, text_sample: str, config: dict,
            similar_context: list[dict] = None) -> dict | None:
    """Route analysis to the configured LLM provider."""
    provider = get_llm_provider(config)
    if provider == "ollama":
        cfg = config["ollama"]
        return analyze_document(file_meta, text_sample,
                                endpoint=cfg["endpoint"],
                                model=cfg["model"],
                                timeout=cfg.get("timeout", 120),
                                similar_context=similar_context)
    elif provider == "openai":
        cfg = config["openai"]
        return analyze_document_openai(file_meta, text_sample,
                                       base_url=cfg["base_url"],
                                       model=cfg["model"],
                                       api_key_env=cfg.get("api_key_env", "OPENAI_API_KEY"),
                                       timeout=cfg.get("timeout", 60),
                                       similar_context=similar_context)
    else:
        logger.warning("No LLM provider configured")
        return None
