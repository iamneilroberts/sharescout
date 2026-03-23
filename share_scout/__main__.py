"""CLI entry point: python -m share_scout"""

import argparse
import logging
import sys

from .config import load_config, load_scoring_rules, apply_cli_overrides


def main():
    parser = argparse.ArgumentParser(
        prog="share_scout",
        description="ShareScout — Network share document discovery tool",
    )
    parser.add_argument(
        "--config", default="config.yaml", help="Path to config.yaml"
    )
    parser.add_argument(
        "--rules", default="scoring_rules.yaml", help="Path to scoring_rules.yaml"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    sub = parser.add_subparsers(dest="command", required=True)

    # crawl subcommand
    crawl_parser = sub.add_parser("crawl", help="Run the document crawler")
    crawl_parser.add_argument("--root-path", help="Root path to crawl")
    crawl_parser.add_argument("--dry-run", action="store_true", help="Score only, don't extract or analyze")
    crawl_parser.add_argument("--ollama-endpoint", help="Ollama API endpoint")
    crawl_parser.add_argument("--ollama-model", help="Ollama model name")
    crawl_parser.add_argument("--db-path", help="SQLite database path")
    crawl_parser.add_argument("--batch-size", type=int, help="Batch commit size")
    crawl_parser.add_argument("--openai-base-url", help="OpenAI-compatible API base URL")
    crawl_parser.add_argument("--openai-model", help="OpenAI-compatible model name")
    crawl_parser.add_argument("--openai-api-key-env", help="Env var name containing API key (default: OPENAI_API_KEY)")
    crawl_parser.add_argument("--domain", help="Domain preset name (e.g., general, policies-and-procedures, developer-workspace, scripts-collection)")
    crawl_parser.add_argument("--max-context-tokens", type=int, help="Override LLM context window size in tokens")

    # web subcommand
    web_parser = sub.add_parser("web", help="Start the web UI")
    web_parser.add_argument("--host", help="Web server host")
    web_parser.add_argument("--port", type=int, help="Web server port")
    web_parser.add_argument("--db-path", help="SQLite database path")

    # embed subcommand
    embed_parser = sub.add_parser("embed", help="Generate embeddings for analyzed catalog files")
    embed_parser.add_argument("--db-path", help="SQLite database path")
    embed_parser.add_argument("--ollama-endpoint", help="Ollama API endpoint")
    embed_parser.add_argument("--embedding-model", help="Ollama embedding model name")

    # claude-proxy subcommand
    proxy_parser = sub.add_parser(
        "claude-proxy",
        help="Start an Ollama-compatible proxy that uses Claude Code CLI for inference",
    )
    proxy_parser.add_argument("--port", type=int, default=11435, help="Port to listen on (default: 11435)")
    proxy_parser.add_argument(
        "--claude-model", default="haiku",
        help="Claude model to use (default: haiku — cheapest, appropriate for simulating 7B)",
    )
    proxy_parser.add_argument(
        "--real", action="store_true",
        help="Real mode: use full Claude quality instead of simulating Mistral 7B",
    )

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    config = load_config(args.config)

    if args.command == "crawl":
        config = apply_cli_overrides(
            config,
            root_path=args.root_path,
            ollama_endpoint=args.ollama_endpoint,
            ollama_model=args.ollama_model,
            db_path=args.db_path,
            batch_size=args.batch_size,
            openai_base_url=args.openai_base_url,
            openai_model=args.openai_model,
            openai_api_key_env=args.openai_api_key_env,
            domain=getattr(args, "domain", None),
            max_context_tokens=getattr(args, "max_context_tokens", None),
            verbose=args.verbose,
        )
        rules = load_scoring_rules(args.rules, config=config)
        from .pipeline import run_crawl
        run_crawl(config, rules, dry_run=args.dry_run)

    elif args.command == "web":
        config = apply_cli_overrides(
            config,
            db_path=args.db_path,
            host=args.host,
            port=args.port,
        )
        from .web.app import create_app
        app = create_app(config)
        app.run(
            host=config["web"]["host"],
            port=config["web"]["port"],
            debug=True,
        )

    elif args.command == "embed":
        if args.db_path:
            config["catalog"]["db_path"] = args.db_path
        if args.ollama_endpoint:
            config["ollama"]["endpoint"] = args.ollama_endpoint
        if args.embedding_model:
            config.setdefault("ollama", {})["embedding_model"] = args.embedding_model
        from .embedder import run_embed
        run_embed(config)

    elif args.command == "claude-proxy":
        from .claude_proxy import run_proxy
        run_proxy(
            port=args.port,
            claude_model=args.claude_model,
            verbose=args.verbose,
            real_mode=args.real,
        )


if __name__ == "__main__":
    main()
