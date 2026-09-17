"""Pytest configuration for notebook tests."""

from pathlib import Path

# Files that require the tools extra (kwcoco, click) in addition to maite
_NOTEBOOK_FILES = {
    "test_notebooks.py",
}


def pytest_ignore_collect(collection_path: Path) -> bool | None:
    """Skip notebook tests when dependencies are not installed."""
    # All tests in this directory require jupytext
    if collection_path.name in _NOTEBOOK_FILES:
        try:
            import jupytext  # noqa: F401
        except ImportError:
            return True

    return None
