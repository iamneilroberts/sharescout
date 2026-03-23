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
MAX_DISTANCE = 0.90



SYSTEM_MESSAGE_QUERY = (
    "You select relevant sentences from numbered text. "
    "Output ONLY the numbers of relevant sentences, like: 1, 5, 12. "
    "If none are relevant, output: NONE"
)

SYSTEM_MESSAGE_CHAT = (
    "You are a document search assistant. You answer questions using the provided document excerpts. "
    "You may use your own knowledge to explain or contextualize, but clearly distinguish "
    "between what comes from the documents and what is your own knowledge. "
    "Respond in English."
)

QUERY_RULES = (
    "Below are numbered sentences extracted from documents. "
    "Output ONLY the numbers of sentences that answer the question. "
    "Format: just comma-separated numbers, nothing else. "
    "If no sentence answers the question, output: NONE\n\n"
)

CHAT_RULES = (
    "## RULES\n"
    "- Prefer information from the document excerpts below.\n"
    "- You may add context from your own knowledge, but mark it clearly as '[my knowledge]'.\n"
    "- Cite the source filename for facts from the excerpts.\n\n"
)


def ask(config: dict, catalog, question: str, top_k: int = 5,
        history: list[dict] = None, mode: str = "query") -> dict:
    """Answer a question using RAG over the document catalog.

    mode: "query" (strict, excerpts only) or "chat" (excerpts + model knowledge)
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
            "mode": mode,
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
    # In query mode, number each sentence for extractive selection.
    # In chat mode, pass raw excerpts for freeform answering.
    import re as _re

    if mode == "query":
        # Split excerpts into numbered sentences with source tracking
        numbered_sentences = []  # (sentence_num, source_idx, sentence_text)
        sentence_num = 0
        for i, source in enumerate(results):
            chunk_text = source.get("chunk_text") or ""
            # Split on sentence boundaries (., !, ?) or newlines
            sentences = _re.split(r'(?<=[.!?])\s+|\n+', chunk_text)
            for sent in sentences:
                sent = sent.strip()
                if len(sent) < 10:  # skip tiny fragments
                    continue
                if sent.startswith('#'):  # skip markdown headers
                    continue
                if _re.match(r'^[-*]\s*$', sent):  # skip empty list items
                    continue
                if _re.match(r'^[\w-]+:\s*\S+$', sent):  # skip frontmatter (key: value)
                    continue
                if sent.startswith('<') and sent.endswith('/>'):  # skip XML/JSX tags
                    continue
                sentence_num += 1
                numbered_sentences.append((sentence_num, i, sent))

        numbered_text = "\n".join(
            f"[{num}] {sent}" for num, _, sent in numbered_sentences
        )
        excerpts_text = numbered_text
        rules = QUERY_RULES
        user_message = (
            f"{rules}"
            f"## Numbered Sentences\n\n{numbered_text}\n\n"
            f"## Question\n{question}"
        )
    else:
        excerpt_blocks = []
        for i, source in enumerate(results, start=1):
            filename = source.get("filename", "unknown")
            path = source.get("path", "")
            chunk_text = source.get("chunk_text") or ""
            excerpt_blocks.append(
                f"### Source {i}: {filename} (path: {path})\n{chunk_text}"
            )
        excerpts_text = "\n\n".join(excerpt_blocks)
        rules = CHAT_RULES
        user_message = (
            f"{rules}"
            f"## Document Excerpts\n\n{excerpts_text}\n\n"
            f"## Question\n{question}"
        )

    # 5. Build messages with conversation history
    system_msg = SYSTEM_MESSAGE_QUERY if mode == "query" else SYSTEM_MESSAGE_CHAT
    messages = [{"role": "system", "content": system_msg}]
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
    raw_answer = _chat(config, messages)
    # Strip leading non-ASCII garbage tokens (gemma3 CJK/Bengali leak)
    import re

    # In query mode, convert selected sentence numbers back to quoted text
    if mode == "query" and numbered_sentences:
        debug["raw_llm_response"] = raw_answer
        raw_answer = re.sub(r'^[^\x00-\x7F]+[,.\s]*', '', raw_answer).strip()
        # Parse numbers from LLM response — only accept numbers in valid range
        max_num = len(numbered_sentences)
        selected_nums = set()
        for match in re.finditer(r'\b(\d+)\b', raw_answer):
            n = int(match.group())
            if 1 <= n <= max_num:
                selected_nums.add(n)
        # Cap at 5 selections to keep answers focused
        if len(selected_nums) > 5:
            selected_nums = set(sorted(selected_nums)[:5])

        if not selected_nums or raw_answer.upper().startswith("NONE"):
            answer = "The indexed documents don't contain information about this."
        else:
            # Reconstruct answer from selected sentences with source attribution
            quote_lines = []
            for num, src_idx, sent in numbered_sentences:
                if num in selected_nums:
                    filename = results[src_idx].get("filename", "unknown")
                    quote_lines.append(f"> {sent}\n> — *{filename}*")
            answer = "\n\n".join(quote_lines) if quote_lines else \
                "The indexed documents don't contain information about this."
    else:
        answer = raw_answer
    answer = re.sub(r'^[^\x00-\x7F]+[,.\s]*', '', answer).strip()
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
