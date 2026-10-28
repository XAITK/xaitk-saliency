"""End-to-end checks for the import guard that only hold across a whole process.

This runs a script from ``tests/scripts/guard/`` in a *fresh* interpreter, because the
claim means nothing inside an interpreter that other tests have already imported
half of xaitk_saliency into.

Everything else about the guard is checked in-process by ``tests/test_guard.py``. A
subprocess is warranted only where import order is the thing under test.

The script is a real ``.py`` file rather than source passed to ``python -c``, so that
ruff and pyright see it. It prints one line, which is what this test asserts on.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "tests" / "scripts" / "guard"


def run(script: str) -> str:
    """Run ``tests/scripts/guard/<script>.py`` in a fresh interpreter, returning its last line."""
    proc = subprocess.run(  # noqa: S603 - fixed interpreter, path built from a literal, no shell
        [sys.executable, str(SCRIPTS / f"{script}.py")],
        capture_output=True,
        text=True,
        cwd=REPO,
        check=False,
    )
    assert proc.returncode == 0, f"{script} failed:\n{proc.stderr}"
    return proc.stdout.strip().splitlines()[-1]


@pytest.mark.core
def test_every_package_imports_without_extras() -> None:
    """The DSO compliance walk imports every package, so none may need an extra.

    A private ``__init__`` that re-exports an extras-dependent leaf fails here,
    which is why the guard always points at leaves.
    """
    assert run("packages_import_without_extras") == "all packages import"
