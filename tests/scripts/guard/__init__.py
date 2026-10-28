"""Scripts run in a fresh interpreter by ``tests/test_import_guards_e2e.py``.

Each module here is a standalone script, not a test. They live as real files
rather than source passed to ``python -c`` so that ruff and pyright see them, and
so a failure points at a line you can open. Every one prints a single line that
the calling test compares against.
"""
