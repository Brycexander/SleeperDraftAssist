from __future__ import annotations

import os
from pathlib import Path


def cache_directory() -> Path:
    """Return a writable cache path for both source and installed deployments."""
    configured = os.environ.get("SLEEPER_CACHE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / ".cache"
