# ollama-sim

Drop-in Ollama replacement powered by Claude Code CLI. Runs a local HTTP server that accepts standard Ollama API requests and forwards them to `claude` for inference. Any app that talks to Ollama works unchanged.

## Why

When you don't have a local GPU or Ollama installed, `ollama-sim` lets you test LLM-dependent workflows using your existing Claude Code authentication. It simulates the behavior of smaller models (concise responses, JSON-only output) so your tests reflect real-world local LLM usage.

## Quick Start

```bash
# Start on default Ollama port, simulating mistral:7b
python tools/ollama-sim.py

# Point your app at it
export OLLAMA_HOST=http://localhost:11434
```

## Options

```
--port PORT           Port to listen on (default: 11434)
--host HOST           Bind address (default: 127.0.0.1)
--sim-model MODEL     Model to simulate (default: mistral:7b)
--claude-model MODEL  Claude model for inference (default: haiku)
--unconstrained       Full Claude capability, no small-model constraints
--verbose             Log prompt/response details
--list-models         Show available model profiles
```

## Model Profiles

Each profile sets a context window size, parameter count, and system prompt that constrains Claude to behave like that model.

| Profile      | Params | Context  | Family  |
|-------------|--------|----------|---------|
| mistral:7b  | 7B     | 8,192    | mistral |
| llama3:8b   | 8B     | 8,192    | llama   |
| llama3.2    | 3B     | 131,072  | llama   |
| gemma2:9b   | 9B     | 8,192    | gemma2  |
| phi3:mini   | 3.8B   | 4,096    | phi3    |
| qwen2:7b    | 7B     | 32,768   | qwen2   |

Unknown model names use a generic 7B/4096-context profile.

## API Coverage

| Endpoint           | Method | Behavior |
|-------------------|--------|----------|
| `/api/chat`       | POST   | Chat completion via Claude CLI |
| `/api/generate`   | POST   | Text generation via Claude CLI |
| `/api/tags`       | GET    | Lists simulated model + all profiles |
| `/api/show`       | POST   | Returns model metadata (context window, params) |
| `/api/ps`         | GET    | Shows "running" simulated model |
| `/api/pull`       | POST   | No-op (instant success) |
| `/api/copy`       | POST   | No-op |
| `/api/delete`     | DELETE | No-op |
| `/`               | GET    | Returns "Ollama is running" |
| `/sim/status`     | GET    | Simulator stats (request count, uptime) |

## Examples

```bash
# Simulate a different model
python tools/ollama-sim.py --sim-model llama3:8b

# Use a more capable Claude model
python tools/ollama-sim.py --claude-model sonnet

# Full Claude power, no behavior constraints
python tools/ollama-sim.py --unconstrained

# Run on a non-standard port alongside real Ollama
python tools/ollama-sim.py --port 11435

# Use with ShareScout
python tools/ollama-sim.py --port 11435 &
python -m share_scout crawl --ollama-endpoint http://localhost:11435
```

## Requirements

- Python 3.10+
- `flask` (`pip install flask`)
- `claude` CLI in PATH (`npm install -g @anthropic-ai/claude-code`)
