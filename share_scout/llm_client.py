"""LLM client for document summarization and categorization."""

import base64
import json
import logging

import httpx
import ollama

from .config import get_llm_provider
from .prompts import (
    DEFAULT_ANALYSIS_PROMPT, get_prompt, get_categories, get_summary_length,
)

logger = logging.getLogger(__name__)

# Kept for backward compat — modules can still reference it
ANALYSIS_PROMPT = DEFAULT_ANALYSIS_PROMPT

# Cache for context budget (computed once per crawl)
_context_budget_cache = {}


def detect_context_budget(config: dict) -> int:
    """Returns available chars for text content after prompt overhead.

    Checks (in order):
    1. config crawl.max_context_tokens override
    2. Ollama model info (context window from model metadata)
    3. Default fallback (4096 tokens)

    Returns character count (tokens * 3.5 chars/token approximation).
    """
    # Config override always wins
    max_tokens = config.get("crawl", {}).get("max_context_tokens")
    if max_tokens:
        prompt_reserve = 800  # tokens for prompt template
        response_reserve = 500  # tokens for JSON response
        available = max_tokens - prompt_reserve - response_reserve
        return int(max(available, 500) * 3.5)

    provider = get_llm_provider(config)

    if provider == "ollama":
        endpoint = config["ollama"]["endpoint"]
        model = config["ollama"]["model"]
        cache_key = f"{endpoint}:{model}"

        if cache_key in _context_budget_cache:
            return _context_budget_cache[cache_key]

        try:
            client = ollama.Client(host=endpoint, timeout=10)
            info = client.show(model)

            # Parse context window from model info
            context_length = None

            # Try modelinfo / model_info dict
            model_info = info.get("model_info", {}) or info.get("modelinfo", {})
            if model_info:
                for key, value in model_info.items():
                    if "context_length" in key.lower():
                        context_length = int(value)
                        break

            # Try parameters string
            if not context_length:
                params = info.get("parameters", "")
                if params and "num_ctx" in params:
                    for line in params.split("\n"):
                        if "num_ctx" in line:
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                try:
                                    context_length = int(parts[-1])
                                except ValueError:
                                    pass
                            break

            if not context_length:
                context_length = 4096  # conservative default

            prompt_reserve = 800
            response_reserve = 500
            available = context_length - prompt_reserve - response_reserve
            budget = int(max(available, 500) * 3.5)

            _context_budget_cache[cache_key] = budget
            logger.info(
                "Context budget: model %s has %d token context → %d chars available",
                model, context_length, budget,
            )
            return budget

        except Exception as e:
            logger.warning("Could not detect context window for %s: %s — using default", model, e)

    # Default: assume 4096 token context
    default_budget = int((4096 - 1300) * 3.5)
    return default_budget


def analyze_document(file_meta: dict, text_sample: str,
                     endpoint: str = "http://localhost:11434",
                     model: str = "llama3.2",
                     timeout: int = 120,
                     similar_context: list[dict] = None,
                     prompt_template: str = None,
                     categories: str = None,
                     summary_length: str = None) -> dict | None:
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

    template = prompt_template or ANALYSIS_PROMPT
    from .prompts import DEFAULT_CATEGORIES
    cats = categories or DEFAULT_CATEGORIES
    length = summary_length or "2-3 sentences"

    prompt = template.format(
        filename=file_meta.get("filename", ""),
        path=file_meta.get("path", ""),
        extension=file_meta.get("extension", ""),
        size_bytes=file_meta.get("size_bytes", 0),
        text_sample=text_sample,
        context_block=context_block,
        summary_length=length,
        categories=cats,
    )

    try:
        client = ollama.Client(host=endpoint, timeout=timeout)
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )

        content = response["message"]["content"].strip()
        result = _parse_json_response(content)
        if not result:
            return None

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
                             similar_context: list[dict] = None,
                             prompt_template: str = None,
                             categories: str = None,
                             summary_length: str = None) -> dict | None:
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

    template = prompt_template or ANALYSIS_PROMPT
    from .prompts import DEFAULT_CATEGORIES
    cats = categories or DEFAULT_CATEGORIES
    length = summary_length or "2-3 sentences"

    prompt = template.format(
        filename=file_meta.get("filename", ""),
        path=file_meta.get("path", ""),
        extension=file_meta.get("extension", ""),
        size_bytes=file_meta.get("size_bytes", 0),
        text_sample=text_sample,
        context_block=context_block,
        summary_length=length,
        categories=cats,
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
        result = _parse_json_response(content)
        if not result:
            return None

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


def caption_image(image_data: bytes, prompt: str, config: dict,
                  vision_config: dict = None) -> str | None:
    """Caption an image using a vision-capable model.

    Args:
        image_data: Raw image bytes
        prompt: Caption prompt
        config: Main config dict
        vision_config: {model, provider, endpoint} or None to use defaults

    Returns caption string or None on failure.
    """
    vision_config = vision_config or {}
    provider = vision_config.get("provider") or get_llm_provider(config)
    vision_model = vision_config.get("model")

    if not vision_model:
        return None

    b64_image = base64.b64encode(image_data).decode("utf-8")

    if provider == "ollama":
        endpoint = vision_config.get("endpoint") or config.get("ollama", {}).get("endpoint", "http://localhost:11434")
        timeout = config.get("ollama", {}).get("timeout", 120)
        try:
            client = ollama.Client(host=endpoint, timeout=timeout)
            response = client.chat(
                model=vision_model,
                messages=[{
                    "role": "user",
                    "content": prompt,
                    "images": [b64_image],
                }],
            )
            return response["message"]["content"].strip()
        except Exception as e:
            logger.warning("Vision captioning failed (ollama/%s): %s", vision_model, e)
            return None

    elif provider == "openai":
        import os
        openai_cfg = config.get("openai", {})
        api_key = os.environ.get(openai_cfg.get("api_key_env", "OPENAI_API_KEY"))
        base_url = vision_config.get("endpoint") or openai_cfg.get("base_url", "")
        if not api_key or not base_url:
            return None
        try:
            response = httpx.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": vision_model,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
                        ],
                    }],
                },
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.warning("Vision captioning failed (openai/%s): %s", vision_model, e)
            return None

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


def check_embedding_model(config: dict) -> bool:
    """Check if the configured embedding model is available.

    Returns False if no embedding_model is configured or if the endpoint
    is unreachable / the model is not found.
    """
    ollama_cfg = config.get("ollama", {})
    embedding_model = ollama_cfg.get("embedding_model")
    if not embedding_model:
        return False

    endpoint = ollama_cfg.get("embedding_endpoint") or ollama_cfg.get("endpoint", "http://localhost:11434")
    try:
        client = ollama.Client(host=endpoint, timeout=5)
        models = client.list()
        model_names = [m["model"] for m in models.get("models", [])]
        # Match with or without :latest tag
        return (embedding_model in model_names
                or f"{embedding_model}:latest" in model_names)
    except Exception:
        return False


def generate_embedding(config: dict, text: str) -> list[float] | None:
    """Generate an embedding vector for the given text using Ollama.

    Uses embedding_endpoint if set, otherwise falls back to ollama.endpoint.
    Returns a list of floats, or None on failure.
    """
    ollama_cfg = config.get("ollama", {})
    embedding_model = ollama_cfg.get("embedding_model")
    if not embedding_model:
        logger.warning("No embedding_model configured — cannot generate embedding")
        return None

    endpoint = ollama_cfg.get("embedding_endpoint") or ollama_cfg.get("endpoint", "http://localhost:11434")
    timeout = ollama_cfg.get("timeout", 120)
    try:
        client = ollama.Client(host=endpoint, timeout=timeout)
        response = client.embed(model=embedding_model, input=text)
        return response["embeddings"][0]
    except Exception as e:
        logger.warning("Embedding generation failed (%s): %s", embedding_model, e)
        return None


def analyze(file_meta: dict, text_sample: str, config: dict,
            similar_context: list[dict] = None,
            prompt_template: str = None,
            categories: str = None,
            summary_length: str = None) -> dict | None:
    """Route analysis to the configured LLM provider."""
    provider = get_llm_provider(config)
    if provider == "ollama":
        cfg = config["ollama"]
        return analyze_document(file_meta, text_sample,
                                endpoint=cfg["endpoint"],
                                model=cfg["model"],
                                timeout=cfg.get("timeout", 120),
                                similar_context=similar_context,
                                prompt_template=prompt_template,
                                categories=categories,
                                summary_length=summary_length)
    elif provider == "openai":
        cfg = config["openai"]
        return analyze_document_openai(file_meta, text_sample,
                                       base_url=cfg["base_url"],
                                       model=cfg["model"],
                                       api_key_env=cfg.get("api_key_env", "OPENAI_API_KEY"),
                                       timeout=cfg.get("timeout", 60),
                                       similar_context=similar_context,
                                       prompt_template=prompt_template,
                                       categories=categories,
                                       summary_length=summary_length)
    else:
        logger.warning("No LLM provider configured")
        return None


def _parse_json_response(content: str) -> dict | None:
    """Extract and parse JSON from LLM response, handling markdown fences."""
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    try:
        result = json.loads(content)
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse LLM JSON response: %s", e)
        return None

    if not all(k in result for k in ("summary", "keywords", "category")):
        logger.warning("LLM response missing expected keys: %s", content[:200])
        return None

    if isinstance(result["keywords"], str):
        result["keywords"] = [k.strip() for k in result["keywords"].split(",")]

    return result
