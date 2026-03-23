"""RAG (Retrieval-Augmented Generation) query support."""

import logging
import os

import httpx
import ollama

from .config import get_llm_provider
from .llm_client import generate_embedding

logger = logging.getLogger(__name__)

# Cosine distance threshold — filter out sources above this (lower = more similar)
# 0.0 = identical, 1.0 = orthogonal, 2.0 = opposite
MAX_DISTANCE = 0.95



SYSTEM_MESSAGE = (
    "You are a document search assistant. You answer questions using ONLY the provided document excerpts. "
    "You must respond in English."
)


def ask(config: dict, catalog, question: str, top_k: int = 5,
        history: list[dict] = None) -> dict:
    """Answer a question using RAG over the document catalog.

    Returns dict with: answer, sources, and debug (timing + pipeline details).
    """
    import time
    debug = {
        "steps": [],
        "timings": {},
        "config": {
            "embedding_model": config.get("ollama", {}).get("embedding_model", "?"),
            "embedding_endpoint": config.get("ollama", {}).get("embedding_endpoint")
                or config.get("ollama", {}).get("endpoint", "?"),
            "chat_model": config.get("ollama", {}).get("model", "?"),
            "chat_endpoint": config.get("ollama", {}).get("endpoint", "?"),
            "chat_provider": get_llm_provider(config),
            "top_k": top_k,
            "max_distance": MAX_DISTANCE,
            "history_turns": len(history) if history else 0,
        },
    }
    t_total = time.time()

    # 1. Embed the question
    t0 = time.time()
    query_vec = generate_embedding(config, question)
    embed_ms = round((time.time() - t0) * 1000)
    debug["timings"]["embed_question_ms"] = embed_ms

    if query_vec is None:
        debug["steps"].append(f"1. Embed question: FAILED ({embed_ms}ms)")
        return {
            "answer": "Unable to process your question: embedding generation failed. "
                      "Check that an embedding model is configured and reachable.",
            "sources": [], "debug": debug,
        }
    debug["steps"].append(f"1. Embed question: OK ({embed_ms}ms, {len(query_vec)} dims)")
    debug["embedding_dims"] = len(query_vec)

    # 2. KNN search
    t0 = time.time()
    raw_results = catalog.vector_search(query_vec, limit=top_k)
    search_ms = round((time.time() - t0) * 1000)
    debug["timings"]["vector_search_ms"] = search_ms
    debug["raw_results"] = len(raw_results)
    debug["raw_distances"] = [round(r.get("distance", 0), 4) for r in raw_results]

    # Filter out poor matches
    results = [r for r in raw_results if r.get("distance", 999) < MAX_DISTANCE]
    debug["filtered_results"] = len(results)
    debug["steps"].append(
        f"2. Vector search: {len(raw_results)} raw → {len(results)} after filter "
        f"(threshold {MAX_DISTANCE}) ({search_ms}ms)"
    )

    if not results:
        debug["timings"]["total_ms"] = round((time.time() - t_total) * 1000)
        debug["steps"].append("3. No relevant results — stopping")
        return {
            "answer": "No relevant documents were found in the catalog for your question.",
            "sources": [], "debug": debug,
        }

    # 3. Resolve chunk text for sources that don't have it
    for source in results:
        if not source.get("chunk_text"):
            analysis = catalog.get_analysis(source["file_id"])
            source["chunk_text"] = analysis["text_sample"] if analysis else ""

    # 4. Build prompt with excerpts
    excerpt_blocks = []
    for i, source in enumerate(results, start=1):
        filename = source.get("filename", "unknown")
        path = source.get("path", "")
        chunk_text = source.get("chunk_text") or ""
        excerpt_blocks.append(
            f"### Source {i}: {filename} (path: {path})\n{chunk_text}"
        )

    excerpts_text = "\n\n".join(excerpt_blocks)
    user_message = (
        f"## IMPORTANT RULES\n"
        f"- Answer ONLY using the document excerpts below. Do NOT use your own knowledge.\n"
        f"- If the excerpts don't contain the answer, say: 'The indexed documents don't contain information about this.'\n"
        f"- Cite the source filename for each fact.\n"
        f"- Be concise and factual.\n\n"
        f"## Document Excerpts\n\n{excerpts_text}\n\n"
        f"## Question\n{question}"
    )

    # 5. Build messages with conversation history
    messages = [{"role": "system", "content": SYSTEM_MESSAGE}]
    if history:
        for turn in history[-6:]:
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_message})

    total_prompt_chars = sum(len(m["content"]) for m in messages)
    debug["prompt_chars"] = total_prompt_chars
    debug["prompt_messages"] = len(messages)
    debug["excerpt_chars"] = len(excerpts_text)
    debug["steps"].append(
        f"3. Build prompt: {len(messages)} messages, {total_prompt_chars:,} chars "
        f"({len(excerpts_text):,} chars excerpts)"
    )

    # 6. Call chat model
    t0 = time.time()
    answer = _chat(config, messages)
    chat_ms = round((time.time() - t0) * 1000)
    debug["timings"]["chat_ms"] = chat_ms
    debug["answer_chars"] = len(answer)
    debug["steps"].append(f"4. LLM chat: {len(answer):,} chars response ({chat_ms}ms)")

    # 7. Build sources list
    sources = [
        {
            "file_id": s.get("file_id"),
            "filename": s.get("filename"),
            "path": s.get("path"),
            "chunk_text": s.get("chunk_text"),
            "distance": s.get("distance"),
        }
        for s in results
    ]

    debug["timings"]["total_ms"] = round((time.time() - t_total) * 1000)
    debug["steps"].append(f"5. Total: {debug['timings']['total_ms']}ms")

    return {"answer": answer, "sources": sources, "debug": debug}


def _chat(config: dict, messages: list[dict]) -> str:
    """Send a chat request to the configured LLM provider.

    Returns the response text, or an error message string on failure.
    """
    provider = get_llm_provider(config)

    if provider == "ollama":
        return _chat_ollama(config, messages)
    elif provider == "openai":
        return _chat_openai(config, messages)
    else:
        logger.warning("RAG: no LLM provider configured")
        return "Unable to generate an answer: no LLM provider is configured."


def _chat_ollama(config: dict, messages: list[dict]) -> str:
    """Chat via Ollama."""
    cfg = config.get("ollama", {})
    endpoint = cfg.get("endpoint", "http://localhost:11434")
    model = cfg.get("model", "llama3.2")
    timeout = cfg.get("timeout", 120)

    try:
        client = ollama.Client(host=endpoint, timeout=timeout)
        response = client.chat(model=model, messages=messages)
        return response["message"]["content"].strip()
    except Exception as e:
        logger.warning("RAG: Ollama chat failed: %s", e)
        return f"Unable to generate an answer: LLM request failed ({e})."


def _chat_openai(config: dict, messages: list[dict]) -> str:
    """Chat via OpenAI-compatible API."""
    cfg = config.get("openai", {})
    base_url = cfg.get("base_url", "").rstrip("/")
    model = cfg.get("model", "")
    timeout = cfg.get("timeout", 60)
    api_key_env = cfg.get("api_key_env", "OPENAI_API_KEY")
    api_key = os.environ.get(api_key_env)

    if not api_key:
        logger.warning("RAG: API key env var %s not set", api_key_env)
        return f"Unable to generate an answer: API key not set ({api_key_env})."

    if not base_url:
        logger.warning("RAG: OpenAI base_url not configured")
        return "Unable to generate an answer: OpenAI base_url not configured."

    try:
        response = httpx.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={"model": model, "messages": messages},
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except httpx.HTTPStatusError as e:
        logger.warning("RAG: OpenAI API error %d: %s", e.response.status_code, e.response.text[:200])
        return f"Unable to generate an answer: API returned HTTP {e.response.status_code}."
    except httpx.TimeoutException:
        logger.warning("RAG: OpenAI API timeout after %ds", timeout)
        return f"Unable to generate an answer: request timed out after {timeout}s."
    except Exception as e:
        logger.warning("RAG: OpenAI chat failed: %s", e)
        return f"Unable to generate an answer: LLM request failed ({e})."
