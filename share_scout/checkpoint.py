"""Checkpoint/resume state for long crawls."""

import logging

from .catalog import Catalog

logger = logging.getLogger(__name__)


class CheckpointManager:
    def __init__(self, catalog: Catalog):
        self.catalog = catalog
        self.run_id = None
        self.files_found = 0
        self.files_analyzed = 0

    def start_run(self, root_path: str) -> int:
        self.run_id = self.catalog.start_crawl_run(root_path)
        logger.info("Started crawl run #%d for %s", self.run_id, root_path)
        return self.run_id

    def should_skip(self, path: str) -> bool:
        """Check if file was already processed in a previous run."""
        return self.catalog.file_exists(path)

    def record_batch(self, found: int = 0, analyzed: int = 0):
        """Update counters after a batch commit."""
        self.files_found += found
        self.files_analyzed += analyzed
        if self.run_id:
            self.catalog.update_crawl_run(
                self.run_id,
                files_found=self.files_found,
                files_analyzed=self.files_analyzed,
            )

    def complete(self):
        if self.run_id:
            self.catalog.complete_crawl_run(
                self.run_id, self.files_found, self.files_analyzed
            )
            logger.info(
                "Crawl run #%d completed: %d found, %d analyzed",
                self.run_id, self.files_found, self.files_analyzed,
            )
