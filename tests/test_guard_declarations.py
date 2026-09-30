"""Checks over what every module *declares* to :func:`xaitk_saliency._guard.guard`.

``tests/test_guard.py`` owns the guard's mechanics. This module owns the other
half: that each module declares the right thing, and that whatever it declared
actually holds in the environment the tests are running in.

That second part is why there is no dependency mocking here. A guarded symbol is
either importable in this environment or it is not, and the guard has to behave
correctly either way. Asserting that invariant means the ``core`` environment
exercises every "extra is missing" path and the ``optional`` environment
exercises every "extra is present" path, over every guarded module, without
evicting anything from ``sys.modules``. The single-extra environments run only
the canaries, so a partially-satisfied multi-extra group (maite installed, tools
absent) is exercised at the code-path level here, never as a live environment
state.
"""

from __future__ import annotations

import ast
import importlib
import importlib.metadata
import pathlib
import re
from collections import defaultdict
from typing import TYPE_CHECKING, cast

import pytest

if TYPE_CHECKING:
    from types import ModuleType

    from xaitk_saliency._guard import Extras, Group, _Guard

SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "xaitk_saliency"
PYPROJECT = SRC.parent.parent / "pyproject.toml"


def _calls_guard(path: pathlib.Path) -> bool:
    """Whether *path* installs a guard, i.e. calls ``guard(...)`` at module level."""
    return any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "guard"
        for node in ast.walk(ast.parse(path.read_text()))
    )


def _guarded_modules() -> list[str]:
    """Every module that installs a guard, by dotted name.

    Discovery is by ``guard()`` call rather than by package, because
    call sites can be plain modules rather than ``__init__.py`` files.
    """
    names = []
    for path in sorted(SRC.rglob("*.py")):
        if not _calls_guard(path):
            continue
        relative = path.parent if path.name == "__init__.py" else path.with_suffix("")
        names.append(".".join(("xaitk_saliency", *relative.relative_to(SRC).parts)))
    return names


def _guard_of(module: str) -> _Guard:
    """The guard *module* installed, reached through the hook it published.

    ``guard()`` hands back ``_Guard.getattr`` as a bound method and the module binds
    it as ``__getattr__``, so the instance is reachable from the module itself. That
    is deliberately the only route: a registry inside ``xaitk_saliency._guard`` would
    be state the package carries purely for these tests, and would need clearing
    every time a test installed a guard over a throwaway namespace.

    Importing is always safe here -- a guarded module has to import with no extras
    installed. ``test_import_guards_e2e.py`` pins that for every package; the
    plain-module call sites, which its walk never imports, are pinned by this
    file's own run in the ``core`` environment.
    """
    imported = importlib.import_module(module)
    hook = imported.__getattr__  # type: ignore[attr-defined]
    return cast("_Guard", hook.__self__)


GUARDED = _guarded_modules()
DECLARED = [(module, group) for module in GUARDED for group in _guard_of(module)._groups]


@pytest.mark.core
def test_every_module_installing_the_hook_is_in_the_guarded_set() -> None:
    """``GUARDED`` must not silently shrink.

    Discovery matches the literal call ``guard(...)``, so a future call through an
    alias or a wrapper would quietly drop its module from every test in this file.
    The ``__getattr__`` assignment anchors the derivation independently: binding
    the hook is required for a guard to function however the call is spelled.
    """
    hooked = []
    for path in sorted(SRC.rglob("*.py")):
        targets = {
            name.id
            for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.Assign)
            for target in node.targets
            for name in (target.elts if isinstance(target, ast.Tuple) else [target])
            if isinstance(name, ast.Name)
        }
        if "__getattr__" in targets:
            relative = path.parent if path.name == "__init__.py" else path.with_suffix("")
            hooked.append(".".join(("xaitk_saliency", *relative.relative_to(SRC).parts)))

    assert hooked == GUARDED


def _terms(extras: Extras) -> set[str]:
    """Flatten a requirement into the set of extra names it mentions."""
    names: set[str] = set()
    for term in extras:
        names.update({term} if isinstance(term, str) else set(term))
    return names


def _normalize_name(name: str) -> str:
    """Normalize extra names per PEP 685."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _type_checking_names(module: str) -> set[str]:
    """Names bound by the ``if TYPE_CHECKING:`` block of *module*, ignoring submodules."""
    path = SRC.parent / pathlib.Path(*module.split("."))
    source = (path / "__init__.py") if path.is_dir() else path.with_suffix(".py")
    names: set[str] = set()
    for node in ast.walk(ast.parse(source.read_text())):
        if not (isinstance(node, ast.If) and getattr(node.test, "id", None) == "TYPE_CHECKING"):
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.ImportFrom):
                names.update(alias.asname or alias.name for alias in child.names)
    return names


@pytest.mark.core
def test_every_declared_extra_exists() -> None:
    """A typo in ``extras=`` would otherwise surface only as advice nobody can follow."""
    known = {
        _normalize_name(name) for name in importlib.metadata.metadata("xaitk-saliency").get_all("Provides-Extra") or []
    }
    unknown = sorted(
        f"{module}: {name!r}"
        for module, group in DECLARED
        for name in sorted(_terms(group.extras))
        if _normalize_name(name) not in known
    )
    assert not unknown, "extras named in a Group but absent from pyproject:\n" + "\n".join(unknown)


@pytest.mark.core
def test_no_symbol_is_claimed_by_two_groups() -> None:
    """Binding re-homes a class globally, so two groups claiming one name is a bug.

    The owner index would silently let the last group win, which means the symbol
    could resolve under a requirement it does not actually have, and ``__all__``
    would list it twice.
    """
    duplicates = []
    for module in GUARDED:
        seen: set[str] = set()
        for group in _guard_of(module)._groups:
            duplicates += [f"{module}.{name}" for name in group.symbols if name in seen]
            seen.update(group.symbols)
    assert not duplicates, f"symbols declared in more than one group: {duplicates}"


@pytest.mark.core
def test_no_leaf_symbol_is_published_by_two_modules() -> None:
    """Two modules cannot publish the same class, because re-homing is global.

    ``_bind`` sets ``__module__`` on the class object itself, so whichever module
    binds last wins and the other silently starts recording a path that does not
    resolve back to it. Different classes that merely share a *name* are fine --
    the collision is on the (leaf module, symbol) pair, which is one class.
    """
    owners: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
    for module in GUARDED:
        for group in _guard_of(module)._groups:
            for name, source in group.symbols.items():
                owners[source, name].append(module)

    shared = {leaf: modules for leaf, modules in owners.items() if len(modules) > 1}
    assert not shared, "a class may be published from exactly one module, but:\n" + "\n".join(
        f"  {source}.{name} is claimed by {modules}" for (source, name), modules in sorted(shared.items())
    )


@pytest.mark.core
@pytest.mark.parametrize("module", GUARDED, ids=GUARDED)
def test_type_checking_block_matches_the_declaration(module: str) -> None:
    """Every guarded symbol needs a ``TYPE_CHECKING`` import, and vice versa.

    The guard resolves names at runtime, which pyright cannot follow, so a symbol
    missing from the block is invisible to type checkers until something uses it.
    A leftover in the block is a symbol that was renamed or dropped.
    """
    module_guard = _guard_of(module)
    declared = {name for group in module_guard._groups for name in group.symbols}
    # Submodules are imported normally, so they may appear in the block without being symbols.
    annotated = _type_checking_names(module) - set(module_guard._submodules)

    assert annotated == declared, (
        f"{module}: TYPE_CHECKING imports and guard symbols disagree.\n"
        f"  declared but not annotated: {sorted(declared - annotated)}\n"
        f"  annotated but not declared: {sorted(annotated - declared)}"
    )


@pytest.mark.core
@pytest.mark.parametrize("module", GUARDED, ids=GUARDED)
def test_declaration_holds_in_this_environment(module: str) -> None:
    """Each declared symbol either resolves, or explains the extra it needs.

    Whichever branch a given environment takes, the two must never both hold and
    never both fail: a name that is advertised has to be reachable, and a name
    that is not has to say why.
    """
    imported = importlib.import_module(module)
    module_guard = _guard_of(module)

    for group in module_guard._groups:
        for name in group.symbols:
            _assert_symbol_is_consistent(module=module, imported=imported, name=name, group=group)


def _assert_symbol_is_consistent(*, module: str, imported: ModuleType, name: str, group: Group) -> None:
    """Assert one symbol is either reachable and re-homed, or refused with useful advice."""
    advertised = name in dir(imported)
    assert advertised == (name in imported.__all__), (
        f"{module}.{name}: __dir__ and __all__ disagree about whether it is available"
    )

    if advertised:
        resolved = getattr(imported, name)
        assert getattr(resolved, "__module__", module) == module, (
            f"{module}.{name} was not re-homed; a serialized config would record its private path"
        )
        return

    # Not advertised, so asking for it must explain itself rather than raising AttributeError.
    with pytest.raises(ImportError) as excinfo:
        getattr(imported, name)
    message = str(excinfo.value)
    assert group.extras, f"{module}.{name} is unavailable but declares no extras, so its failure is a bug"
    missing = sorted(term for term in _terms(group.extras) if term not in message)
    assert not missing, f"{module}.{name}: error message does not name {missing}\n  message: {message}"


# --------------------------------------------------------------------- no self-imports


def _is_submodule(*, package: str, name: str) -> bool:
    """Whether ``package.name`` is a module on disk rather than a guard-bound name."""
    dotted = f"{package}.{name}"
    candidate = SRC.parent / dotted.replace(".", "/")
    return candidate.with_suffix(".py").exists() or candidate.is_dir()


def _self_imports(*, path: pathlib.Path) -> list[str]:
    """Report every bound name *path* imports from the package that contains it.

    ``from . import X`` is the same import spelled relatively, so it is matched too.
    """
    relative = str(path.relative_to(SRC)).removesuffix(".py")
    dotted = "xaitk_saliency." + relative.replace("/", ".")
    package = dotted.rsplit(".", maxsplit=1)[0]
    offenders = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and (node.module == package or (node.level == 1 and node.module is None)):
            offenders += [
                f"{path.relative_to(SRC.parent)}:{node.lineno} imports {alias.name!r} from "
                f"{package!r}; import it from its leaf module instead"
                for alias in node.names
                if not _is_submodule(package=package, name=alias.name)
            ]
    return offenders


@pytest.mark.core
def test_no_module_imports_a_bound_name_from_its_own_package() -> None:
    """A module must not import a guard-bound name from the package that contains it.

    ``from xaitk_saliency.interop.maite.object_detection import MAITEDetectionTarget``
    inside ``_coco_maite_object_detection_dataset.py`` reaches back through a
    package that is still executing its own ``guard()`` call, so it resolves only if
    that name happens to have been bound already. That makes the order of a
    ``symbols`` dict load-bearing: reordering it, or inserting an entry above the
    wrong line, breaks the import with a confusing circular-import error.
    Naming the leaf module directly carries no such dependency.

    Importing a *submodule* from the parent package stays fine -- the import system
    resolves those regardless of what has been bound.
    """
    offenders: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        if path.name != "__init__.py":
            offenders += _self_imports(path=path)

    assert not offenders, "\n".join(offenders)
