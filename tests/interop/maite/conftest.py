"""Pytest configuration for MAITE interop tests."""

import importlib
from pathlib import Path

# Mapping of test filenames to extra modules they require beyond maite
_FILE_DEPENDENCIES: dict[str, list[str]] = {
    "test_sal_on_coco_dets.py": ["click"],
}


def _can_import(module_name: str) -> bool:
    """Check if a module can be imported."""
    try:
        importlib.import_module(module_name)
    except ImportError:
        return False
    return True


def pytest_ignore_collect(collection_path: Path) -> bool | None:
    """Skip MAITE tests when the maite extra (or a per-file extra) is not installed."""
    if not _can_import("maite"):
        return True

    for dep in _FILE_DEPENDENCIES.get(collection_path.name, []):
        if not _can_import(dep):
            return True

    return None
