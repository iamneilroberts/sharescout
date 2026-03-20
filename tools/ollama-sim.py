#!/usr/bin/env python3
"""
ollama-sim — A general-purpose Ollama API simulator powered by Claude Code CLI.

Drop-in replacement for a local Ollama server. Any application that talks to
Ollama can point at this simulator and get responses via Claude (haiku by default).

Usage:
    python ollama-sim.py                          # port 11434, simulates mistral:7b
    python ollama-sim.py --port 11435             # custom port
    python ollama-sim.py --sim-model llama3:8b    # simulate a different model
    python ollama-sim.py --claude-model sonnet    # use a different Claude model
    python ollama-sim.py --unconstrained          # skip the "act like a small model" prompt
    python ollama-sim.py --verbose                # log prompts and responses

Then configure your app:
    ollama.endpoint = "http://localhost:11434"     # or whatever port you chose

Implements:
    POST /api/chat          Chat completions (non-streaming)
    POST /api/generate      Text generation (non-streaming)
    GET  /api/tags          List available models
    POST /api/show          Model metadata
    GET  /api/ps            Running models
    POST /api/pull          Simulate pull (instant no-op)
    POST /api/copy          Simulate copy (instant no-op)
    DELETE /api/delete       Simulate delete (instant no-op)
    GET  /                  Health check
    HEAD /                  Health check

Requirements:
    - Python 3.10+
    - flask (pip install flask)
    - claude CLI in PATH (npm install -g @anthropic-ai/claude-code)
"""

import argparse
import json
import logging
import subprocess
import sys
import time

try:
    from flask import Flask, request, jsonify, Response
except ImportError:
    print("Flask is required: pip install flask")
    sys.exit(1)

logger = logging.getLogger("ollama-sim")

# ── Model Profiles ──
# Simulate different model behaviors via context window size and system prompt.
# Add new profiles here to simulate other models.

MODEL_PROFILES = {
    "mistral:7b": {
        "context_length": 8192,
        "parameter_size": "7B",
        "family": "mistral",
        "quantization": "Q4_0",
        "system_hint": (
            "You are simulating a Mistral 7B model. "
            "Keep responses concise and direct. "
            "Stick strictly to JSON when asked for JSON — no markdown fences, no preamble. "
            "Focus on the most prominent information. "
            "Don't over-elaborate or use overly sophisticated vocabulary."
        ),
    },
    "llama3:8b": {
        "context_length": 8192,
        "parameter_size": "8B",
        "family": "llama",
        "quantization": "Q4_0",
        "system_hint": (
            "You are simulating a Llama 3 8B model. "
            "Keep responses helpful but concise. "
            "Stick strictly to JSON when asked for JSON — no markdown fences, no preamble. "
            "Provide clear, practical answers."
        ),
    },
    "llama3.2": {
        "context_length": 131072,
        "parameter_size": "3B",
        "family": "llama",
        "quantization": "Q4_K_M",
        "system_hint": (
            "You are simulating a Llama 3.2 3B model. "
            "Keep responses short and to the point. "
            "Stick strictly to JSON when asked for JSON — no markdown fences, no preamble. "
            "You may miss subtle nuances. Focus on the obvious."
        ),
    },
    "gemma2:9b": {
        "context_length": 8192,
        "parameter_size": "9B",
        "family": "gemma2",
        "quantization": "Q4_0",
        "system_hint": (
            "You are simulating a Gemma 2 9B model. "
            "Provide clear, well-structured responses. "
            "Stick strictly to JSON when asked for JSON — no markdown fences, no preamble."
        ),
    },
    "phi3:mini": {
        "context_length": 4096,
        "parameter_size": "3.8B",
        "family": "phi3",
        "quantization": "Q4_0",
        "system_hint": (
            "You are simulating a Phi-3 Mini 3.8B model. "
            "Keep responses brief. Simpler vocabulary. "
            "Stick strictly to JSON when asked for JSON — no markdown fences, no preamble. "
            "Occasionally miss details a larger model would catch."
        ),
    },
    "qwen2:7b": {
        "context_length": 32768,
        "parameter_size": "7B",
        "family": "qwen2",
        "quantization": "Q4_0",
        "system_hint": (
            "You are simulating a Qwen2 7B model. "
            "Respond concisely and accurately. "
            "Stick strictly to JSON when asked for JSON — no markdown fences, no preamble."
        ),
    },
}

# Fallback for unknown model names
DEFAULT_PROFILE = {
    "context_length": 4096,
    "parameter_size": "7B",
    "family": "unknown",
    "quantization": "Q4_0",
    "system_hint": (
        "Keep responses concise and direct. "
        "Stick strictly to JSON when asked for JSON — no markdown fences, no preamble."
    ),
}


class OllamaSim:
    """Ollama-compatible HTTP server backed by Claude Code CLI."""

    def __init__(self, sim_model="mistral:7b", claude_model="haiku",
                 unconstrained=False, verbose=False):
        self.sim_model = sim_model
        self.claude_model = claude_model
        self.unconstrained = unconstrained
        self.verbose = verbose
        self.profile = MODEL_PROFILES.get(sim_model, DEFAULT_PROFILE)
        self.request_count = 0
        self.start_time = time.time()

    def _build_system_prompt(self):
        """Build the system prompt for Claude CLI."""
        if self.unconstrained:
            return (
                "Respond directly to the user's request. "
                "When asked for JSON output, respond with ONLY valid JSON — "
                "no markdown fences, no explanation, no preamble."
            )
        return (
            f"{self.profile['system_hint']} "
            "Never mention that you are Claude or simulating another model. "
            "When asked for JSON, respond with ONLY the raw JSON object."
        )

    def call_claude(self, prompt: str) -> tuple[str, float]:
        """Call claude CLI and return (response_text, elapsed_seconds)."""
        system_prompt = self._build_system_prompt()

        cmd = [
            "claude",
            "--print",
            "--model", self.claude_model,
            "--output-format", "text",
            "--system-prompt", system_prompt,
        ]

        if self.verbose:
            logger.info("→ claude %s | prompt: %d chars", self.claude_model, len(prompt))

        start = time.time()
        try:
            result = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=300,
            )
            elapsed = time.time() - start

            if result.returncode != 0:
                stderr = result.stderr.strip()
                logger.warning("claude CLI error (rc=%d): %s", result.returncode, stderr[:500])
                return "", elapsed

            response = result.stdout.strip()

            if self.verbose:
                logger.info("← %d chars in %.1fs", len(response), elapsed)

            return response, elapsed

        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            logger.warning("claude CLI timed out after %.0fs", elapsed)
            return "", elapsed
        except FileNotFoundError:
            logger.error("'claude' CLI not found in PATH")
            return "", 0
        except Exception as e:
            elapsed = time.time() - start
            logger.warning("claude CLI failed: %s", e)
            return "", elapsed

    def _make_chat_response(self, model, content, elapsed):
        """Format a response in Ollama chat format."""
        elapsed_ns = int(elapsed * 1e9)
        return {
            "model": model,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            "message": {"role": "assistant", "content": content},
            "done": True,
            "done_reason": "stop",
            "total_duration": elapsed_ns,
            "load_duration": 0,
            "prompt_eval_count": 0,
            "prompt_eval_duration": 0,
            "eval_count": max(1, len(content.split())),
            "eval_duration": elapsed_ns,
        }

    def _make_generate_response(self, model, content, elapsed):
        """Format a response in Ollama generate format."""
        elapsed_ns = int(elapsed * 1e9)
        return {
            "model": model,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            "response": content,
            "done": True,
            "done_reason": "stop",
            "context": [],
            "total_duration": elapsed_ns,
            "load_duration": 0,
            "prompt_eval_count": 0,
            "prompt_eval_duration": 0,
            "eval_count": max(1, len(content.split())),
            "eval_duration": elapsed_ns,
        }

    def _model_info(self, model_name=None):
        """Return model metadata in Ollama format."""
        name = model_name or self.sim_model
        profile = MODEL_PROFILES.get(name, self.profile)
        return {
            "name": name,
            "model": name,
            "modified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "size": 4_000_000_000,
            "digest": f"sim-{name.replace(':', '-')}",
            "details": {
                "parent_model": "",
                "format": "gguf",
                "family": profile["family"],
                "families": [profile["family"]],
                "parameter_size": profile["parameter_size"],
                "quantization_level": profile["quantization"],
            },
        }


def create_app(sim: OllamaSim) -> Flask:
    """Create the Ollama-compatible Flask application."""
    app = Flask(__name__)

    # ── Chat ──

    @app.route("/api/chat", methods=["POST"])
    def api_chat():
        data = request.get_json(silent=True) or {}
        messages = data.get("messages", [])
        model = data.get("model", sim.sim_model)
        sim.request_count += 1

        # Build a single prompt from the message history
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"[System: {content}]")
            elif role == "assistant":
                parts.append(f"[Assistant: {content}]")
            else:
                parts.append(content)

        prompt = "\n\n".join(parts)
        if not prompt:
            return jsonify({"error": "no messages provided"}), 400

        response_text, elapsed = sim.call_claude(prompt)
        if not response_text:
            return jsonify({"error": "claude CLI returned empty response"}), 500

        return jsonify(sim._make_chat_response(model, response_text, elapsed))

    # ── Generate ──

    @app.route("/api/generate", methods=["POST"])
    def api_generate():
        data = request.get_json(silent=True) or {}
        prompt = data.get("prompt", "")
        model = data.get("model", sim.sim_model)
        sim.request_count += 1

        if not prompt:
            return jsonify({"error": "no prompt provided"}), 400

        # Prepend system prompt if provided
        system = data.get("system", "")
        if system:
            prompt = f"[System: {system}]\n\n{prompt}"

        response_text, elapsed = sim.call_claude(prompt)
        if not response_text:
            return jsonify({"error": "claude CLI returned empty response"}), 500

        return jsonify(sim._make_generate_response(model, response_text, elapsed))

    # ── List Models ──

    @app.route("/api/tags", methods=["GET"])
    @app.route("/api/tags/", methods=["GET"])
    def api_tags():
        models = [sim._model_info()]
        # Also list other known profiles
        for name in MODEL_PROFILES:
            if name != sim.sim_model:
                info = sim._model_info(name)
                models.append(info)
        return jsonify({"models": models})

    # ── Show Model ──

    @app.route("/api/show", methods=["POST"])
    def api_show():
        data = request.get_json(silent=True) or {}
        name = data.get("name", data.get("model", sim.sim_model))
        profile = MODEL_PROFILES.get(name, sim.profile)
        return jsonify({
            "modelfile": f'FROM {name}\nPARAMETER num_ctx {profile["context_length"]}',
            "parameters": f'num_ctx {profile["context_length"]}',
            "template": "{{ .Prompt }}",
            "model_info": {
                "general.architecture": profile["family"],
                "general.context_length": profile["context_length"],
                "general.parameter_count": _parse_param_count(profile["parameter_size"]),
                "general.quantization_version": 2,
            },
            "details": {
                "parent_model": "",
                "format": "gguf",
                "family": profile["family"],
                "families": [profile["family"]],
                "parameter_size": profile["parameter_size"],
                "quantization_level": profile["quantization"],
            },
        })

    # ── Running Models ──

    @app.route("/api/ps", methods=["GET"])
    def api_ps():
        uptime_min = (time.time() - sim.start_time) / 60
        profile = sim.profile
        return jsonify({
            "models": [{
                "name": sim.sim_model,
                "model": sim.sim_model,
                "size": 4_000_000_000,
                "size_vram": 4_000_000_000,
                "digest": f"sim-{sim.sim_model.replace(':', '-')}",
                "details": {
                    "parent_model": "",
                    "format": "gguf",
                    "family": profile["family"],
                    "families": [profile["family"]],
                    "parameter_size": profile["parameter_size"],
                    "quantization_level": profile["quantization"],
                },
                "context_length": profile["context_length"],
                "expires_at": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ",
                    time.gmtime(time.time() + 300),
                ),
            }]
        })

    # ── Model Management (no-ops) ──

    @app.route("/api/pull", methods=["POST"])
    def api_pull():
        data = request.get_json(silent=True) or {}
        name = data.get("name", sim.sim_model)
        logger.info("Simulated pull: %s (instant)", name)
        return jsonify({"status": "success"})

    @app.route("/api/copy", methods=["POST"])
    def api_copy():
        data = request.get_json(silent=True) or {}
        logger.info("Simulated copy: %s → %s", data.get("source"), data.get("destination"))
        return Response(status=200)

    @app.route("/api/delete", methods=["DELETE"])
    def api_delete():
        data = request.get_json(silent=True) or {}
        logger.info("Simulated delete: %s", data.get("name"))
        return Response(status=200)

    # ── Health ──

    @app.route("/", methods=["GET", "HEAD"])
    def health():
        return "Ollama is running"

    @app.route("/api/version", methods=["GET"])
    def version():
        return jsonify({"version": "0.6.2"})

    # ── Stats (bonus, not part of Ollama API) ──

    @app.route("/sim/status", methods=["GET"])
    def sim_status():
        uptime = time.time() - sim.start_time
        return jsonify({
            "simulator": "ollama-sim",
            "sim_model": sim.sim_model,
            "claude_model": sim.claude_model,
            "unconstrained": sim.unconstrained,
            "requests_handled": sim.request_count,
            "uptime_seconds": round(uptime),
            "uptime_human": f"{uptime/3600:.1f}h" if uptime > 3600 else f"{uptime/60:.1f}m",
        })

    return app


def _parse_param_count(size_str: str) -> int:
    """Convert '7B' → 7000000000."""
    s = size_str.upper().replace(" ", "")
    if s.endswith("B"):
        try:
            return int(float(s[:-1]) * 1_000_000_000)
        except ValueError:
            pass
    return 7_000_000_000


def main():
    parser = argparse.ArgumentParser(
        prog="ollama-sim",
        description="Ollama API simulator powered by Claude Code CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  ollama-sim                                  Start on port 11434, simulate mistral:7b\n"
            "  ollama-sim --port 11435                     Custom port\n"
            "  ollama-sim --sim-model llama3:8b            Simulate Llama 3 8B\n"
            "  ollama-sim --claude-model sonnet            Use Claude Sonnet instead of Haiku\n"
            "  ollama-sim --unconstrained                  Full Claude capability (no 7B simulation)\n"
            "  ollama-sim --list-models                    Show available model profiles\n"
            "\n"
            "Known model profiles: " + ", ".join(sorted(MODEL_PROFILES.keys())) + "\n"
            "Any other model name will use a generic profile.\n"
        ),
    )
    parser.add_argument("--port", type=int, default=11434, help="Port to listen on (default: 11434)")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--sim-model", default="mistral:7b",
                        help="Model name to simulate (default: mistral:7b)")
    parser.add_argument("--claude-model", default="haiku",
                        help="Claude model for inference (default: haiku)")
    parser.add_argument("--unconstrained", action="store_true",
                        help="Don't constrain Claude to simulate a smaller model")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Log prompts and responses")
    parser.add_argument("--list-models", action="store_true",
                        help="List known model profiles and exit")

    args = parser.parse_args()

    if args.list_models:
        print("\nKnown model profiles:")
        print(f"{'Model':<20} {'Params':<10} {'Context':<10} {'Family'}")
        print("-" * 55)
        for name, p in sorted(MODEL_PROFILES.items()):
            print(f"{name:<20} {p['parameter_size']:<10} {p['context_length']:<10} {p['family']}")
        print(f"\nAny other model name uses a generic 7B/4096-context profile.")
        return

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Verify claude CLI is available
    try:
        result = subprocess.run(
            ["claude", "--version"],
            capture_output=True, text=True, timeout=10,
        )
        claude_version = result.stdout.strip()
        logger.info("Claude CLI: %s", claude_version)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("Error: 'claude' CLI not found. Install it first:")
        print("  npm install -g @anthropic-ai/claude-code")
        sys.exit(1)

    sim = OllamaSim(
        sim_model=args.sim_model,
        claude_model=args.claude_model,
        unconstrained=args.unconstrained,
        verbose=args.verbose,
    )
    app = create_app(sim)

    profile = sim.profile
    constrained = "unconstrained" if args.unconstrained else f"simulating {args.sim_model}"

    print(f"")
    print(f"  ollama-sim — Ollama API simulator")
    print(f"  ─────────────────────────────────────────────")
    print(f"  Backend:     Claude ({args.claude_model})")
    print(f"  Simulating:  {args.sim_model} ({profile['parameter_size']}, {profile['context_length']} ctx)")
    print(f"  Mode:        {constrained}")
    print(f"  Listening:   http://{args.host}:{args.port}")
    print(f"")
    print(f"  Point your app at this endpoint:")
    print(f"    OLLAMA_HOST=http://{args.host}:{args.port}")
    print(f"")
    print(f"  Ctrl+C to stop")
    print(f"")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
