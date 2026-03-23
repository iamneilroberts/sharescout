"""RAG (Retrieval-Augmented Generation) query support."""

import logging
import os

import httpx
import ollama

from .config import get_llm_provider
from .llm_client import generate_embedding

logger = logging.getLogger(__name__)


def ask(config: dict, catalog, question: str, top_k: int = 5) -> dict:
    """Answer a question using RAG over the document catalog.

    Steps:
    1. Embed the question
    2. KNN search for relevant chunks
    3. Build a prompt with retrieved excerpts
    4. Call the chat model
    5. Return answer + sources

    Returns:
        {
            "answer": str,
            "sources": [{"file_id", "filename", "path", "chunk_text", "distance"}]
        }

    On failure, "answer" will be an error message and "sources" will be empty.
    """
    # 1. Embed the question
    query_vec = generate_embedding(config, question)
    if query_vec is None:
        logger.warning("RAG: embedding generation failed — cannot answer question")
        return {
            "answer": "Unable to process your question: embedding generation failed. "
                      "Check that an embedding model is configured and reachable.",
            "sources": [],
        }

    # 2. KNN search
    results = catalog.vector_search(query_vec, limit=top_k)
    if not results:
        logger.info("RAG: no relevant documents found for question")
        return {
            "answer": "No relevant documents were found in the catalog for your question.",
            "sources": [],
        }

    # 3. Resolve chunk text for sources that don't have it
    for source in results:
        if not source.get("chunk_text"):
            analysis = catalog.get_analysis(source["file_id"])
            source["chunk_text"] = analysis["text_sample"] if analysis else ""

    # 4. Build prompt
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
        f"## Document Excerpts\n\n{excerpts_text}\n\n"
        f"## Question\n{question}"
    )

    system_message = (
        "Answer the user's question based on the document excerpts below. "
        "Cite sources by filename. "
        "If the excerpts don't contain relevant information, say so."
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    # 5. Call chat model
    answer = _chat(config, messages)

    # 6. Build sources list
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

    return {"answer": answer, "sources": sources}


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
