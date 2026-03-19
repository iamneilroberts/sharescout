"""Load and manage application configuration."""

import os
from pathlib import Path

import yaml


DEFAULT_CONFIG = {
    "crawl": {
        "root_path": ".",
        "batch_size": 100,
        "skip_hidden": True,
        "skip_dirs": ["$RECYCLE.BIN", "System Volume Information", ".git", "__pycache__"],
        "text_sample_max_chars": 4000,
        "hash_bytes": 65536,
    },
    "ollama": {
        "endpoint": "http://localhost:11434",
        "model": "llama3.2",
        "timeout": 120,
    },
    "catalog": {
        "db_path": "share_scout.db",
    },
    "web": {
        "host": "0.0.0.0",
        "port": 8080,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge override into base, recursively for nested dicts."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str = "config.yaml") -> dict:
    """Load config from YAML file, merged with defaults."""
    config = DEFAULT_CONFIG.copy()
    path = Path(config_path)
    if path.exists():
        with open(path) as f:
            user_config = yaml.safe_load(f) or {}
        config = _deep_merge(config, user_config)
    return config


def load_scoring_rules(rules_path: str = "scoring_rules.yaml") -> dict:
    """Load scoring rules from YAML file."""
    path = Path(rules_path)
    if not path.exists():
        raise FileNotFoundError(f"Scoring rules file not found: {rules_path}")
    with open(path) as f:
        return yaml.safe_load(f)


def load_project_groups(groups_path: str = "project_groups.yaml") -> dict:
    """Load project grouping config. Returns {group_name: [project_names]}."""
    path = Path(groups_path)
    if not path.exists():
        return {}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return data.get("groups", {})


def apply_cli_overrides(config: dict, **overrides) -> dict:
    """Apply CLI flag overrides to config. None values are ignored."""
    if overrides.get("root_path"):
        config["crawl"]["root_path"] = overrides["root_path"]
    if overrides.get("ollama_endpoint"):
        config["ollama"]["endpoint"] = overrides["ollama_endpoint"]
    if overrides.get("ollama_model"):
        config["ollama"]["model"] = overrides["ollama_model"]
    if overrides.get("db_path"):
        config["catalog"]["db_path"] = overrides["db_path"]
    if overrides.get("batch_size"):
        config["crawl"]["batch_size"] = overrides["batch_size"]
    if overrides.get("openai_base_url") or overrides.get("openai_model"):
        config.setdefault("openai", {})
        if overrides.get("openai_base_url"):
            config["openai"]["base_url"] = overrides["openai_base_url"]
        if overrides.get("openai_model"):
            config["openai"]["model"] = overrides["openai_model"]
        if overrides.get("openai_api_key_env"):
            config["openai"]["api_key_env"] = overrides["openai_api_key_env"]
    if overrides.get("host"):
        config["web"]["host"] = overrides["host"]
    if overrides.get("port"):
        config["web"]["port"] = overrides["port"]
    return config


def get_llm_provider(config: dict) -> str:
    """Determine which LLM provider is configured."""
    if "openai" in config and config["openai"].get("model"):
        api_key_env = config["openai"].get("api_key_env", "OPENAI_API_KEY")
        if os.environ.get(api_key_env):
            return "openai"
    if "ollama" in config:
        return "ollama"
    return "none"
