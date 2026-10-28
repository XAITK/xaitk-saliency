"""Canary tests: if an environment claims a marker, the symbols behind it must import.

``tests/test_guard_declarations.py`` asserts that each declared symbol either resolves
or explains the extra it needs -- deliberately true whichever way an environment is
built. That is exactly why it cannot catch a *misconfigured* one, where a tox
environment or CI job selects ``-m maite`` without the ``maite`` extra actually installed.

These tests close that gap. Each is generated from a ``Group`` that declares extras and
carries the marker for those extras, so selecting a marker asserts the matching symbols
are importable, and failing is a hard ``pytest.fail`` rather than a skip.

Nothing here is edited when adding an implementation: declaring the ``Group`` is what
creates its canary.
"""

from __future__ import annotations

import importlib
import re
from typing import TYPE_CHECKING

import pytest
from _pytest.mark.expression import Expression

from tests.test_guard_declarations import DECLARED, PYPROJECT, _terms

if TYPE_CHECKING:
    from xaitk_saliency._guard import Extras, Group


def _markers_for(extras: Extras) -> list[str]:
    """The marker names an environment would carry to satisfy *extras*.

    Every marker is named after the extra it requires, so no mapping is
    needed here.
    """
    return sorted(_terms(extras))


CANARIES = [
    pytest.param(
        module,
        group,
        marks=[getattr(pytest.mark, marker) for marker in _markers_for(group.extras)],
        id=f"{module}-{'+'.join(_markers_for(group.extras))}",
    )
    # A group with no extras resolves in every environment, so a canary over it would
    # assert nothing about how the environment was built.
    for module, group in DECLARED
    if group.extras
]


def _registered_markers() -> set[str]:
    """The marker names pyproject registers, read line by line to survive comments."""
    registered: set[str] = set()
    in_block = False
    for line in PYPROJECT.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("markers = ["):
            in_block = True
        elif in_block and stripped == "]":
            return registered
        elif in_block and (entry := re.match(r'"([^:"]+)', stripped)):
            registered.add(entry.group(1))
    raise AssertionError("pyproject.toml has no [tool.pytest.ini_options] markers list")


def _selected_by(*, expression: str, marks: set[str]) -> bool:
    """Whether pytest would collect a test carrying *marks* under ``-m`` *expression*."""

    def matcher(name: str, /, **_: object) -> bool:
        return name in marks

    return Expression.compile(expression).evaluate(matcher)


def _tox_expressions() -> list[str]:
    """Every ``-m`` expression a tox command selects, skipping comment lines."""
    lines = (
        line
        for line in (PYPROJECT.parent / "tox.ini").read_text().splitlines()
        if not line.lstrip().startswith(("#", ";"))
    )
    return [match.group(2) for line in lines if (match := re.search(r"""-m\s+(['"])(.+?)\1""", line))]


@pytest.mark.core
def test_every_canary_marker_is_registered_and_selected_somewhere() -> None:
    """A canary only runs if an environment's ``-m`` expression matches its marks.

    Markers are derived from the extras a ``Group`` declares, so declaring a new
    extra mints a marker nothing knows about yet. pytest is not run with strict
    markers and every tox environment hand-lists its ``-m`` expression, so without
    this check the new canary would be collected by nothing and silently never run
    -- the misconfiguration this file exists to catch, one level up.

    Selection is judged with pytest's own expression grammar, not by word:
    ``tools`` appearing only in ``-m "maite and not tools"`` selects nothing. The
    suite-wide ``optional`` mark is deliberately left out of the match:
    ``-m optional`` would match every canary, but ``require_marker`` skips them
    there (see its docstring), so only a dedicated environment proves a canary
    runs.
    """
    registered = _registered_markers()
    expressions = _tox_expressions()
    assert expressions, "no -m expressions found in tox.ini commands"

    problems = []
    for marks in sorted({tuple(_markers_for(group.extras)) for _, group in DECLARED if group.extras}):
        if unregistered := set(marks) - registered:
            problems.append(f"{'+'.join(marks)}: not in pyproject markers: {sorted(unregistered)}")
        if not any(_selected_by(expression=expression, marks=set(marks)) for expression in expressions):
            problems.append(f"{'+'.join(marks)}: no tox -m expression selects a canary carrying these marks")

    assert not problems, "\n".join(problems)


@pytest.mark.parametrize(("module", "group"), CANARIES)
@pytest.mark.usefixtures("require_marker")
def test_declared_symbols_import_when_their_marker_is_selected(module: str, group: Group) -> None:
    """FAIL, rather than skip, when a selected marker's extras are not really installed.

    Reaching a symbol through the module is what the guard makes conditional, so this
    goes through ``getattr`` rather than importing the leaf directly -- the leaf might
    import fine while the guard still refuses to publish the name.
    """
    imported = importlib.import_module(module)

    unreachable = []
    for name in group.symbols:
        try:
            getattr(imported, name)
        except ImportError as ex:  # noqa: PERF203 - one report per symbol, not one per group
            unreachable.append(f"  {name}: {ex}")

    if unreachable:
        markers = ", ".join(_markers_for(group.extras))
        pytest.fail(
            f"Running with the {markers} marker, but {module} cannot publish every symbol "
            f"declared behind {list(group.extras)}. Is the environment missing an extra?\n" + "\n".join(unreachable),
        )
