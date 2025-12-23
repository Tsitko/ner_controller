"""Test helper to ensure src is importable."""

from __future__ import annotations

import sys
from pathlib import Path


def add_src_path() -> None:
    """Add the src directory to sys.path for imports."""
    project_root = Path(__file__).resolve().parents[1]
    src_root = project_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
