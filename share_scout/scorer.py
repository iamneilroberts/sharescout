"""Apply configurable scoring rules to files."""

import fnmatch
import logging

logger = logging.getLogger(__name__)


class Scorer:
    def __init__(self, rules: dict):
        self.threshold = rules.get("score_threshold", 20)

        # Build extension → score lookup
        self._ext_scores = {}
        for group in rules.get("extension_scores", {}).values():
            score = group["score"]
            for ext in group["extensions"]:
                self._ext_scores[ext.lower()] = score

        self._path_rules = rules.get("path_rules", [])
        self._size_rules = rules.get("size_rules", [])

    def score(self, file_meta: dict) -> tuple[float, str | None]:
        """Score a file. Returns (score, skip_reason or None)."""
        total = 0.0

        # Extension score
        ext = file_meta.get("extension", "")
        if ext and ext in self._ext_scores:
            total += self._ext_scores[ext]

        # Path pattern score
        path = file_meta.get("path", "")
        # Normalize to forward slashes for matching
        norm_path = path.replace("\\", "/")
        for rule in self._path_rules:
            if fnmatch.fnmatch(norm_path, rule["pattern"]):
                total += rule["score"]

        # Size score
        size = file_meta.get("size_bytes", 0)
        for rule in self._size_rules:
            min_b = rule.get("min_bytes", 0)
            max_b = rule.get("max_bytes", float("inf"))
            if min_b <= size <= max_b:
                total += rule["score"]

        # Clamp to 0-100 range
        total = max(0, min(100, total))

        skip_reason = None
        if total < self.threshold:
            skip_reason = f"score {total:.0f} below threshold {self.threshold}"

        return total, skip_reason
