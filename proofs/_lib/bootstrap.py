"""Ensure the repository root is on ``sys.path`` for ``import proofs`` / ``buildml``."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_repo_on_path() -> Path:
    """Insert the BuildML repo root at the front of ``sys.path`` if needed."""
    repo_root = Path(__file__).resolve().parents[2]
    root_s = str(repo_root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    return repo_root
