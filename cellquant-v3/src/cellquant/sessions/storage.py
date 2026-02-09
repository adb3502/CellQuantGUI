"""Filesystem layout helpers for session storage."""

from pathlib import Path


def session_layout(base: Path) -> dict:
    """Return the expected directory structure for a session."""
    return {
        "tiles": base / "tiles",
        "thumbnails": base / "thumbnails",
        "masks": base / "masks",
        "results": base / "results",
        "exports": base / "exports",
    }


def ensure_session_dirs(base: Path):
    """Create all session subdirectories."""
    for path in session_layout(base).values():
        path.mkdir(parents=True, exist_ok=True)
