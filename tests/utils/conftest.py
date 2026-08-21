"""Pytest configuration for utils tests."""

import importlib
from pathlib import Path

# Mapping of test filenames to the extra modules they require beyond core
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
    """Skip test files whose required dependencies are not installed."""
    for dep in _FILE_DEPENDENCIES.get(collection_path.name, []):
        if not _can_import(dep):
            return True
    return None
