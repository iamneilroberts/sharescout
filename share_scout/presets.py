"""Load and validate domain preset configurations."""

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def load_preset(name: str, presets_dir: str = "presets") -> dict:
    """Load a domain preset from YAML file.

    Returns the preset dict, or empty dict if not found.
    """
    path = Path(presets_dir) / f"{name}.yaml"
    if not path.exists():
        logger.warning("Preset '%s' not found at %s", name, path)
        return {}
    with open(path) as f:
        preset = yaml.safe_load(f) or {}
    errors = validate_preset(preset)
    if errors:
        logger.warning("Preset '%s' has validation issues: %s", name, "; ".join(errors))
    return preset


def list_presets(presets_dir: str = "presets") -> list[str]:
    """Return names of available presets (without .yaml extension)."""
    path = Path(presets_dir)
    if not path.is_dir():
        return []
    return sorted(p.stem for p in path.glob("*.yaml"))


def validate_preset(preset: dict) -> list[str]:
    """Validate preset structure. Returns list of error messages (empty = valid)."""
    errors = []
    if not preset.get("name"):
        errors.append("missing 'name' field")
    if not preset.get("description"):
        errors.append("missing 'description' field")

    # Validate prompts section if present
    prompts = preset.get("prompts", {})
    if prompts:
        for key in ("analysis", "rollup", "image_caption"):
            if key in prompts and not isinstance(prompts[key], str):
                errors.append(f"prompts.{key} must be a string")

    # Validate scoring_rules if present
    scoring = preset.get("scoring_rules", {})
    if scoring:
        if "extensions" in scoring and not isinstance(scoring["extensions"], list):
            errors.append("scoring_rules.extensions must be a list")
        if "path_rules" in scoring and not isinstance(scoring["path_rules"], list):
            errors.append("scoring_rules.path_rules must be a list")
        if "score_threshold" in scoring and not isinstance(scoring["score_threshold"], (int, float)):
            errors.append("scoring_rules.score_threshold must be a number")

    # Validate categories if present
    if "categories" in preset and not isinstance(preset["categories"], list):
        errors.append("categories must be a list")

    return errors
