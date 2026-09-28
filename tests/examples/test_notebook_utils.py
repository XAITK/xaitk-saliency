"""The following functions are an adapted from MAITE. To see the original implementation, see https://gitlab.jatic.net/jatic/cdao/maite/-/blob/main/src/maite/_internals/testing/pyright.py?ref_type=heads."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal, TypedDict

from typing_extensions import NotRequired

_found_path = shutil.which("pyright")
PYRIGHT_PATH = Path(_found_path) if _found_path else None
del _found_path


@contextmanager
def chdir() -> Generator[Path, Any, Any]:
    with tempfile.TemporaryDirectory() as tmpdirname:
        old_dir = os.getcwd()  # get current working directory (cwd)
        try:
            os.chdir(tmpdirname)  # change cwd to the temp-directory
            yield Path(tmpdirname)  # yields control to the test to be run
        finally:
            os.chdir(old_dir)


docstring_re = re.compile(
    r"""
    # Source consists of a PS1 line followed by zero or more PS2 lines.
    (?P<source>
        (?:^(?P<indent> [ ]*) >>>    .*)    # PS1 line
        (?:\n           [ ]*  \.\.\. .*)*)  # PS2 lines
    \n?
    """,
    re.MULTILINE | re.VERBOSE,
)


class Summary(TypedDict):
    filesAnalyzed: int
    errorCount: int
    warningCount: int
    informationCount: int
    timeInSec: float


class LineInfo(TypedDict):
    line: int
    character: int


class Range(TypedDict):
    start: LineInfo
    end: LineInfo


class Diagnostic(TypedDict):
    file: str
    severity: Literal["error", "warning", "information"]
    message: str
    range: Range
    rule: NotRequired[str]


class PyrightOutput(TypedDict):
    """The schema for the JSON output of a pyright scan."""

    # # doc-ignore: NOQA
    version: str
    time: str
    generalDiagnostics: list[Diagnostic]
    summary: Summary


def notebook_to_py_text(path_to_nb: Path) -> str:
    import jupytext

    ntbk = jupytext.read(path_to_nb, fmt="ipynb")
    return jupytext.writes(ntbk, fmt=".py")


def get_docstring_examples(doc: str) -> str:
    prefix = ">>> "

    # contains input lines of docstring examples with all indentation
    # and REPL markers removed
    src_lines: list[str] = []

    for source, indent in docstring_re.findall(doc):
        source: str
        indent: str
        src_lines = [line[len(indent) + len(prefix) :] for line in source.splitlines()]
        src_lines.append("")  # newline between blocks
    return "\n".join(src_lines)


def _validate_path_to_pyright(path_to_pyright: Path | None) -> None:
    if path_to_pyright is None:  # pragma: no cover
        raise ModuleNotFoundError(
            "`pyright` was not found. It may need to be installed.",
        )
    if not path_to_pyright.is_file():
        raise FileNotFoundError(
            f"`path_to_pyright` – {path_to_pyright} – doesn't exist.",
        )


def _format_outputs(scan: PyrightOutput) -> PyrightOutput:
    out = scan["generalDiagnostics"]
    diagnostics: list[Diagnostic] = []

    if out:
        item = out[0]
        file_str = item["file"]
        if "SCAN_DIR" in file_str:
            name = Path(file_str).name
        else:
            file_path_all = Path(file_str)
            name = file_path_all.name

        diagnostic = item.copy()
        diagnostic["file"] = name
        diagnostics.append(diagnostic)

    severities = Counter(d["severity"] for d in diagnostics)
    summary = Summary(
        filesAnalyzed=1,
        errorCount=severities["error"],
        warningCount=severities["warning"],
        informationCount=severities["information"],
        timeInSec=scan["summary"]["timeInSec"],
    )
    return PyrightOutput(
        version=scan["version"],
        time=scan["time"],
        generalDiagnostics=diagnostics,
        summary=summary,
    )


def pyright_analyze(
    *,
    notebook_path_str: str,
    path_to_pyright: Path | None = PYRIGHT_PATH,
) -> PyrightOutput:
    r"""Scan a Python notebook with pyright.

    Some common pyright configuration options are exposed via this function for
    convenience; a full pyright JSON config can be specified to completely control
    the behavior of pyright.

    This function requires that pyright is installed and can be run from the command
    line [1]_.

    Parameters
    ----------
    *notebook_path_str : str
        A path to a file to a notebook to scan.

    path_to_pyright : Path, optional, keyword-only
        Path to the pyright executable (see installation instructions: [4]_).
        Defaults to `shutil.where('pyright')` if the executable can be found.

    Returns:
    -------
    list[dict[str, Any]]  (In one-to-one correspondence with `code_objs_and_or_paths`)
        The JSON-decoded results of the scan [3]_.
            - version: str
            - time: str
            - generalDiagnostics: list[DiagnosticDict] (one entry per error/warning)
            - summary: SummaryDict

        See Notes for more details.

    Notes:
    -----
    `SummaryDict` consists of:
        - filesAnalyzed: int
        - errorCount: int
        - warningCount: int
        - informationCount: int
        - timeInSec: float

    `DiagnosticDict` consists of:
        - file: str
        - severity: Literal["error", "warning", "information"]
        - message: str
        - range: _Range
        - rule: NotRequired[str]

    References:
    ----------
    .. [1] https://github.com/microsoft/pyright/blob/aad650ec373a9894c6f13490c2950398095829c6/README.md#command-line
    .. [2] https://github.com/microsoft/pyright/blob/main/docs/configuration.md
    .. [3] https://docs.python.org/3/library/doctest.html
    .. [4] https://github.com/microsoft/pyright/blob/main/docs/command-line.md#json-output
    """
    _validate_path_to_pyright(path_to_pyright=path_to_pyright)
    notebook_path = Path(notebook_path_str).resolve()

    if not notebook_path.exists():
        raise FileNotFoundError(
            f"Specified path {notebook_path} does not exist. Cannot be scanned by pyright.",
        )

    if notebook_path.suffix != ".ipynb":
        raise ValueError(
            f"{notebook_path}: Only ipynb is supported by `pyright_analyze`.",
        )

    source = notebook_to_py_text(notebook_path)

    with chdir():
        cwd = Path.cwd()
        file_ = cwd / f"{getattr(notebook_path, '__name__', 'source')}.py"
        file_.write_text(source, encoding="utf-8")

        # Calling subprocess of untrusted input raises ruff error.
        proc = subprocess.run(  # noqa: S603
            [str(path_to_pyright.absolute()), str(cwd.absolute()), "--outputjson"],  # pyright: ignore [reportOptionalMemberAccess]
            cwd=cwd,
            encoding="utf-8",
            text=True,
            capture_output=True,
        )
        try:
            scan: PyrightOutput = json.loads(proc.stdout)
        except Exception as e:  # pragma: no cover
            print(proc.stdout)
            raise e

    return _format_outputs(scan)


def list_error_messages(results: PyrightOutput) -> list[str]:
    """A convenience function that returns a list of error messages reported by pyright.

    Parameters
    ----------
    results : PyrightOutput
        The results of pyright_analyze.

    Returns:
    -------
    list[str]
        A list of error messages.
    """
    # doc-ignore: EX01 SA01 GL01
    return [
        f"(line start) {e['range']['start']['line']}: {e['message']}"
        for e in results["generalDiagnostics"]
        if e["severity"] == "error"
    ]
