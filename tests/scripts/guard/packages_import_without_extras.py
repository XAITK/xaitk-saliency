"""Every package under ``xaitk_saliency`` must import with no optional dependency installed.

This is the DSO compliance walk, which imports every package, so none of them may
need an extra to be importable. It is what forces the guard to point at *leaf*
modules: a private ``__init__`` that re-exports an extras-dependent leaf fails here.

It also protects plugin discovery: smqtk loads every registered entrypoint in a bare
loop, so one unimportable module takes down the whole discovery pass rather than
just its own entry -- importing cleanly is upstream of ``get_impls()`` working at all.

Blocking uses a meta-path finder rather than the absence of the packages, so the
check is real in the all-extras environment too, where they are installed. Prints
``all packages import`` on success, or the list of failures.
"""

import ast
import importlib
import pathlib
import sys
import warnings

SRC = pathlib.Path("src/xaitk_saliency")


def _module_name(path: pathlib.Path) -> str:
    """The dotted name of a file under ``src/xaitk_saliency``."""
    relative = path.parent if path.name == "__init__.py" else path.with_suffix("")
    return ".".join(("xaitk_saliency", *relative.relative_to(SRC).parts))


def _imported_roots(tree: ast.Module) -> set[str]:
    """Top-level names *tree* imports at runtime, e.g. ``kwcoco`` for ``import kwcoco.coco_dataset``.

    ``if TYPE_CHECKING:`` blocks are skipped: those imports never execute, so they
    are no evidence a root must stay importable -- counting one in a non-gated
    module would quietly remove its root from the blocked set.
    """
    type_checking = {
        child
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and getattr(node.test, "id", None) == "TYPE_CHECKING"
        for child in ast.walk(node)
    }
    roots: set[str] = set()
    for node in ast.walk(tree):
        if node in type_checking:
            continue
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def _string_constants(tree: ast.Module) -> dict[str, str]:
    """Module-level ``_NAME = "value"`` assignments, which leaf paths interpolate."""
    constants: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            constants.update({t.id: node.value.value for t in node.targets if isinstance(t, ast.Name)})
    return constants


def _resolve(*, node: ast.expr, constants: dict[str, str]) -> str | None:  # noqa: C901 - one branch per f-string part shape
    """A leaf path written literally or as ``f"{_PREFIX}.leaf"``, or None if unreadable."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if not isinstance(node, ast.JoinedStr):
        return None
    parts = []
    for value in node.values:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            parts.append(value.value)
        elif isinstance(value, ast.FormattedValue) and isinstance(value.value, ast.Name):
            resolved = constants.get(value.value.id)
            if resolved is None:
                return None
            parts.append(resolved)
        else:
            return None
    return "".join(parts)


def _gated_leaves(trees: dict[str, ast.Module]) -> tuple[set[str], list[str]]:  # noqa: C901 - one branch per way a declaration can be unreadable
    """Leaf modules declared by a ``Group`` that names extras, plus anything unreadable.

    A path this cannot read statically would silently shrink the blocked set and make
    the whole check weaker without failing, so it is reported rather than skipped.
    """
    leaves: set[str] = set()
    unreadable: list[str] = []
    for module, tree in trees.items():
        constants = _string_constants(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func_name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if func_name != "Group":
                continue
            if node.args:
                unreadable.append(f"{module}: line {node.lineno}: a Group call uses positional arguments")
                continue
            keywords = {keyword.arg: keyword.value for keyword in node.keywords}
            extras = keywords.get("extras")
            if extras is None:
                continue
            if not isinstance(extras, ast.List):
                unreadable.append(f"{module}: line {extras.lineno}: a Group's extras= is not a list literal")
                continue
            if not extras.elts:
                continue
            symbols = keywords.get("symbols")
            if not isinstance(symbols, ast.Dict):
                unreadable.append(f"{module}: a gated Group's symbols= is not a dict literal")
                continue
            for value in symbols.values:
                if (leaf := _resolve(node=value, constants=constants)) is None:
                    unreadable.append(f"{module}: line {value.lineno}: cannot read a gated leaf path")
                else:
                    leaves.add(leaf)
    return leaves, unreadable


def _xaitk_imports(tree: ast.Module) -> set[str]:
    """The fully qualified ``xaitk_saliency.*`` modules *tree* imports."""
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module)
    return {module for module in modules if module.startswith("xaitk_saliency.")}


def _blocked(trees: dict[str, ast.Module]) -> set[str]:
    """Third-party modules only extras-gated code imports.

    Derived rather than listed, so adding an implementation cannot leave a new
    dependency unblocked and quietly make this check vacuous.

    Reading the ``Group`` declarations is what makes the split possible. Anything a
    non-gated module imports is a required dependency and must stay importable, or
    every package would fail here for the wrong reason.
    """
    leaves, unreadable = _gated_leaves(trees)
    if unreadable:
        raise RuntimeError("\n".join(unreadable))

    gated, stack = set(), list(leaves)
    while stack:  # a gated leaf's private helpers are gated too
        current = stack.pop()
        if current in gated or current not in trees:
            continue
        gated.add(current)
        stack += _xaitk_imports(trees[current])

    def third_party(names: set[str]) -> set[str]:
        return {name for name in names if name != "xaitk_saliency" and name not in sys.stdlib_module_names}

    inside = {root for name in gated for root in third_party(_imported_roots(trees[name]))}
    outside = {root for name, tree in trees.items() if name not in gated for root in third_party(_imported_roots(tree))}
    return inside - outside


TREES = {_module_name(path): ast.parse(path.read_text()) for path in sorted(SRC.rglob("*.py"))}
BLOCKED = _blocked(TREES)


class Blocker:
    """Meta-path finder that makes every package in :data:`BLOCKED` unimportable."""

    def find_spec(self, fullname: str, _path: object = None, _target: object = None) -> None:
        """Raise for a blocked top-level package, otherwise defer to the next finder.

        ``find_spec``, not the legacy ``find_module``: that protocol was removed in
        3.12, so a ``find_module``-based blocker is ignored there and proves nothing.
        """
        root = fullname.split(".")[0]
        if root in BLOCKED:
            raise ModuleNotFoundError(f"No module named {root!r}")
        return None  # noqa: RET501 - the finder protocol wants an explicit "not mine"


def main() -> None:
    """Install the blocker, then import every xaitk_saliency package and report the failures."""
    warnings.simplefilter("ignore")

    if not BLOCKED:
        print("nothing derived to block -- this check would be vacuous")
        return

    sys.meta_path.insert(0, Blocker())

    try:
        importlib.import_module(sorted(BLOCKED)[0])
    except ModuleNotFoundError:
        pass
    else:
        print("blocker is inert -- this check would be vacuous")
        return

    src = pathlib.Path("src/xaitk_saliency")
    names = ["xaitk_saliency"] + sorted(
        ".".join(("xaitk_saliency", *path.parent.relative_to(src).parts))
        for path in src.rglob("__init__.py")
        if path.parent != src
    )
    failed = [report for name in names if (report := _import_failure(name))]
    print(failed or "all packages import")


def _import_failure(name: str) -> str:
    """Import *name*, returning a description of what went wrong or "" if it worked."""
    try:
        importlib.import_module(name)
    except Exception as ex:  # noqa: BLE001 - any failure to import is a failure to report
        return f"{name}: {type(ex).__name__}: {ex}"
    return ""


if __name__ == "__main__":
    main()
