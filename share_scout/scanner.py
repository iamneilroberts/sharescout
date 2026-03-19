"""Walk filesystem and yield file metadata."""

import logging
import os
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def scan_files(root_path: str, skip_hidden: bool = True,
               skip_dirs: list[str] = None) -> "Generator[dict, None, None]":
    """Walk root_path and yield file metadata dicts.

    Streams results — never loads the full file list into memory.
    """
    skip_dirs = set(skip_dirs or [])
    root = Path(root_path)

    if not root.exists():
        logger.error("Root path does not exist: %s", root_path)
        return
    if not root.is_dir():
        logger.error("Root path is not a directory: %s", root_path)
        return

    logger.info("Scanning: %s", root_path)

    for dirpath, dirnames, filenames in os.walk(root):
        # Filter directories in-place to prevent os.walk from descending
        dirnames[:] = [
            d for d in dirnames
            if d not in skip_dirs
            and not (skip_hidden and d.startswith("."))
        ]

        for filename in filenames:
            if skip_hidden and filename.startswith("."):
                continue

            filepath = os.path.join(dirpath, filename)
            try:
                stat = os.stat(filepath)
                ext = Path(filename).suffix.lower()
                yield {
                    "path": filepath,
                    "filename": filename,
                    "extension": ext if ext else None,
                    "size_bytes": stat.st_size,
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                }
            except (OSError, PermissionError) as e:
                logger.warning("Cannot stat %s: %s", filepath, e)
                continue
