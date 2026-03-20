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
    "domain": "general",
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
    """Load config from YAML file, merged with defaults.

    Three-layer merge: defaults < preset < user_config.
    """
    config = DEFAULT_CONFIG.copy()
    path = Path(config_path)
    if path.exists():
        with open(path) as f:
            user_config = yaml.safe_load(f) or {}
        config = _deep_merge(config, user_config)

    # Load domain preset and merge (defaults < preset < user_config)
    domain = config.get("domain", "general")
    if domain:
        from .presets import load_preset
        preset = load_preset(domain)
        if preset:
            # Store the full preset for use by other modules
            config["_preset"] = preset

            # Merge preset LLM settings under the main config
            preset_llm = preset.get("llm", {})
            if preset_llm:
                config.setdefault("llm", {})
                config["llm"] = _deep_merge(config.get("llm", {}), preset_llm)

            # Merge preset extraction settings under crawl
            preset_extraction = preset.get("extraction", {})
            if preset_extraction:
                config["crawl"] = _deep_merge(config["crawl"], preset_extraction)

            # Merge preset skip_dirs into crawl skip_dirs
            preset_skip = preset.get("skip_dirs", [])
            if preset_skip:
                existing = set(config["crawl"].get("skip_dirs", []))
                for d in preset_skip:
                    if d not in existing:
                        config["crawl"]["skip_dirs"].append(d)

        # Re-apply user config on top so user always wins
        if path.exists():
            with open(path) as f:
                user_config = yaml.safe_load(f) or {}
            config = _deep_merge(config, user_config)
            # Preserve the preset reference
            if preset:
                config["_preset"] = preset

    return config


def load_scoring_rules(rules_path: str = "scoring_rules.yaml", config: dict = None) -> dict:
    """Load scoring rules from YAML file.

    If a domain preset is active, merge: default_scoring < preset.scoring_rules < scoring_rules.yaml
    """
    path = Path(rules_path)
    if not path.exists():
        raise FileNotFoundError(f"Scoring rules file not found: {rules_path}")
    with open(path) as f:
        user_rules = yaml.safe_load(f)

    # If we have a preset with scoring_rules, merge them
    if config and "_preset" in config:
        preset_scoring = config["_preset"].get("scoring_rules", {})
        if preset_scoring:
            # Preset scoring provides additional extension rules and path rules
            # that get merged under the user's rules (user wins)
            merged = dict(preset_scoring)
            # User's rules override preset rules
            for key, value in user_rules.items():
                merged[key] = value
            return merged

    return user_rules


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
    if overrides.get("domain"):
        config["domain"] = overrides["domain"]
    if overrides.get("max_context_tokens"):
        config["crawl"]["max_context_tokens"] = overrides["max_context_tokens"]
    if overrides.get("verbose"):
        config["verbose"] = True
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
