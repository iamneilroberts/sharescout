"""Ollama-compatible HTTP proxy that uses Claude Code CLI for inference.

Starts a local server that accepts Ollama API requests and forwards them
to `claude` CLI in print mode. The crawler sees it as a normal Ollama
instance — just point ollama.endpoint at this proxy's address.

Usage:
    python -m share_scout claude-proxy              # default port 11435
    python -m share_scout claude-proxy --port 11435 --claude-model haiku

Then configure ShareScout:
    ollama:
      endpoint: "http://localhost:11435"
      model: "claude-proxy"
"""

import json
import logging
import subprocess
import sys
import time
from flask import Flask, request, jsonify

logger = logging.getLogger(__name__)

# Approximate Mistral 7B behavior constraints
MISTRAL_SYSTEM_PROMPT = """\
You are simulating a local Mistral 7B model for document analysis testing. \
Behave as a competent but not exceptional 7B-parameter language model would:

Constraints to simulate Mistral 7B:
- Keep summaries concise (2-5 sentences max). Don't over-elaborate.
- Stick strictly to JSON output when asked for JSON — no preamble, no explanation.
- Occasionally miss subtle nuances that a larger model would catch.
- Don't use overly sophisticated vocabulary or complex sentence structures.
- Focus on the most obvious/prominent information in the text.
- Keywords should be the most salient terms, not exhaustive.
- Category selection should be straightforward — pick the most obvious fit.
- If the text is ambiguous, pick the simpler interpretation.
- Never mention that you are Claude or simulating another model.

You MUST respond with ONLY valid JSON when the user's prompt asks for JSON. \
No markdown fences, no explanation, no preamble — just the raw JSON object."""


REAL_SYSTEM_PROMPT = "Respond with only valid JSON when asked for JSON. No markdown fences."

REAL_MODE_CONTEXT = 200_000
SIMULATED_MODE_CONTEXT = 8192


def create_proxy_app(claude_model: str = "haiku", verbose: bool = False, real_mode: bool = False) -> Flask:
    """Create the Ollama-compatible proxy Flask app."""
    app = Flask(__name__)

    # Track request count for stats
    app._request_count = 0
    app._start_time = time.time()

    system_prompt = REAL_SYSTEM_PROMPT if real_mode else MISTRAL_SYSTEM_PROMPT
    context_length = REAL_MODE_CONTEXT if real_mode else SIMULATED_MODE_CONTEXT

    def _call_claude(prompt: str, images: list[str] = None) -> str:
        """Call claude CLI in print mode and return the response text."""
        cmd = [
            "claude",
            "--print",
            "--model", claude_model,
            "--output-format", "text",
            "--system-prompt", system_prompt,
        ]

        if verbose:
            logger.info("Calling claude CLI (model=%s, prompt=%d chars)", claude_model, len(prompt))

        try:
            result = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=180,  # 3 minute timeout
            )

            if result.returncode != 0:
                stderr = result.stderr.strip()
                logger.warning("claude CLI error (rc=%d): %s", result.returncode, stderr[:500])
                return ""

            response = result.stdout.strip()
            if verbose:
                logger.info("claude response: %d chars", len(response))
            return response

        except subprocess.TimeoutExpired:
            logger.warning("claude CLI timed out after 180s")
            return ""
        except FileNotFoundError:
            logger.error("'claude' CLI not found in PATH")
            return ""
        except Exception as e:
            logger.warning("claude CLI call failed: %s", e)
            return ""

    # ── Ollama-compatible API endpoints ──

    @app.route("/api/chat", methods=["POST"])
    def api_chat():
        """Ollama chat endpoint — the main analysis route."""
        data = request.get_json(silent=True) or {}
        messages = data.get("messages", [])
        app._request_count += 1

        # Extract the user prompt (last user message)
        prompt = ""
        images = []
        for msg in messages:
            if msg.get("role") == "user":
                prompt = msg.get("content", "")
                images = msg.get("images", [])

        if not prompt:
            return jsonify({"error": "no prompt provided"}), 400

        start = time.time()
        response_text = _call_claude(prompt, images)
        elapsed_ns = int((time.time() - start) * 1e9)

        if not response_text:
            return jsonify({"error": "claude CLI returned empty response"}), 500

        # Return in Ollama chat response format
        return jsonify({
            "model": data.get("model", "claude-proxy"),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "message": {
                "role": "assistant",
                "content": response_text,
            },
            "done": True,
            "total_duration": elapsed_ns,
            "eval_duration": elapsed_ns,
            "prompt_eval_duration": 0,
            "eval_count": len(response_text.split()),
            "prompt_eval_count": len(prompt.split()),
            "load_duration": 0,
        })

    @app.route("/api/tags", methods=["GET"])
    @app.route("/api/tags/", methods=["GET"])
    def api_tags():
        """List available models — returns the proxy model."""
        return jsonify({
            "models": [{
                "name": "claude-proxy",
                "model": "claude-proxy",
                "modified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "size": 0,
                "digest": "claude-proxy",
                "details": {
                    "parent_model": "",
                    "format": "claude",
                    "family": "claude",
                    "families": ["claude"],
                    "parameter_size": "simulated-7B",
                    "quantization_level": "none",
                },
            }]
        })

    @app.route("/api/show", methods=["POST"])
    def api_show():
        """Model info — returns context window based on mode."""
        param_size = "claude" if real_mode else "7B (simulated)"
        return jsonify({
            "modelfile": "",
            "parameters": f"num_ctx {context_length}",
            "model_info": {
                "general.context_length": context_length,
                "general.parameter_count": 7000000000,
            },
            "details": {
                "parent_model": "",
                "format": "claude",
                "family": "claude",
                "parameter_size": param_size,
                "quantization_level": "none",
            },
        })

    @app.route("/api/ps", methods=["GET"])
    def api_ps():
        """Running models — for web UI status display."""
        param_size = "claude" if real_mode else "7B (simulated)"
        return jsonify({
            "models": [{
                "name": f"claude-proxy (via {claude_model})",
                "model": "claude-proxy",
                "size": 0,
                "size_vram": 0,
                "digest": "claude-proxy",
                "details": {
                    "parameter_size": param_size,
                    "quantization_level": "none",
                    "family": "claude",
                },
                "context_length": context_length,
                "expires_at": "",
            }]
        })

    @app.route("/", methods=["GET"])
    def health():
        """Health check — Ollama returns 'Ollama is running'."""
        return "Claude proxy is running"

    @app.route("/api/version", methods=["GET"])
    def version():
        return jsonify({"version": "claude-proxy-1.0"})

    return app


def run_proxy(port: int = 11435, claude_model: str = "haiku", verbose: bool = False, real_mode: bool = False):
    """Start the Ollama-compatible proxy server."""
    # Verify claude CLI is available
    try:
        result = subprocess.run(
            ["claude", "--version"],
            capture_output=True, text=True, timeout=10,
        )
        claude_version = result.stdout.strip()
        logger.info("Found claude CLI: %s", claude_version)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        logger.error("'claude' CLI not found. Install Claude Code first: npm install -g @anthropic-ai/claude-code")
        sys.exit(1)

    app = create_proxy_app(claude_model=claude_model, verbose=verbose, real_mode=real_mode)

    mode_label = "Real mode (full Claude quality)" if real_mode else "Simulated Mistral 7B"

    print(f"")
    print(f"  Claude Proxy — Ollama-compatible server")
    print(f"  Mode:       {mode_label} (via claude {claude_model})")
    print(f"  Listening:  http://localhost:{port}")
    print(f"")
    print(f"  Configure ShareScout to use this proxy:")
    print(f"    ollama:")
    print(f"      endpoint: \"http://localhost:{port}\"")
    print(f"      model: \"claude-proxy\"")
    print(f"")

    app.run(host="127.0.0.1", port=port, debug=False)
