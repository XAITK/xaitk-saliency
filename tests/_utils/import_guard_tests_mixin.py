"""Mixin for testing import guard behavior when dependencies are unavailable.

Python caches every imported module in ``sys.modules``.  Subsequent
``import`` statements return the cached object instead of re-executing
the module's code.  To test what happens when a dependency is *missing*,
we need to:

1. **Evict** the target module (and its submodules / private implementation
   modules) from ``sys.modules`` so that the next ``import`` re-executes
   the module-level code — including the import guard logic we want to test.
2. **Inject** ``None`` for each dependency we want to simulate as missing.
   When ``sys.modules[dep]`` is ``None``, Python raises ``ImportError``
   on ``import dep``, which is how we fake a missing package.
3. **Restore** everything after the test so other tests see the real modules.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import import_module
from types import ModuleType
from typing import Any

import pytest


def _get_module_prefixes(module_path: str) -> list[str]:
    """Build the list of ``sys.modules`` key-prefixes to evict.

    .. note:: **This assumes a project-specific convention** where every
       public module like ``nrtk.impls.perturb_image.photometric.blur``
       has a corresponding private implementation module at
       ``nrtk.impls.perturb_image.photometric._blur``.  The private
       prefix is always generated regardless of whether the private
       module actually exists; this is harmless because evicting a key
       that isn't in ``sys.modules`` is a no-op.

    Both prefixes must be evicted so that re-importing the public module
    also re-executes the private module (where the real import guard
    typically lives).

    Args:
        module_path: Dotted path of the public module
            (e.g. ``"nrtk.impls.perturb_image.photometric.blur"``).

    Returns:
        A list of one or two prefixes.  For the example above:
        ``["nrtk.impls.perturb_image.photometric.blur",
          "nrtk.impls.perturb_image.photometric._blur"]``.
    """
    parts = module_path.rsplit(sep=".", maxsplit=1)
    if len(parts) == 2:
        parent, name = parts
        private_path = f"{parent}._{name}"
        return [module_path, private_path]
    return [module_path]


def _should_reset_module(mod: str, prefixes: list[str]) -> bool:
    """Return True if *mod* matches or is a child of any prefix.

    A module ``"pkg.foo.blur.utils"`` matches prefix ``"pkg.foo.blur"``
    because it is a submodule.  This ensures that the entire subtree
    rooted at each prefix is evicted from ``sys.modules``.
    """
    return any(mod == prefix or mod.startswith(prefix + ".") for prefix in prefixes)


def _clear_modules_by_prefix(prefixes: list[str], saved: dict[str, Any]) -> None:
    """Evict matching modules from ``sys.modules``, saving them for later restoration.

    Each evicted module object is stored in *saved* so it can be put back
    into ``sys.modules`` after the test (see ``_restore_modules``).
    """
    for mod in list(sys.modules.keys()):
        if _should_reset_module(mod=mod, prefixes=prefixes):
            saved[mod] = sys.modules.pop(mod)


def _remove_modules_by_prefix(prefixes: list[str]) -> None:
    """Evict matching modules from ``sys.modules`` without saving them.

    Used during cleanup to discard the "tainted" module objects that were
    imported while dependencies were mocked, before restoring the originals.
    """
    for mod in list(sys.modules.keys()):
        if _should_reset_module(mod=mod, prefixes=prefixes):
            sys.modules.pop(mod, None)


def _save_and_mock_deps(deps_to_mock: list[str], saved: dict[str, Any]) -> None:
    """Save and evict each dep and its submodules, then inject ``None`` sentinels."""
    for dep in deps_to_mock:
        # Save and evict the dep AND all its submodules (e.g. "maite.protocols.*")
        # so that cached submodule entries can't bypass the None sentinel on the parent.
        for mod in list(sys.modules.keys()):
            if mod == dep or mod.startswith(dep + "."):
                saved[mod] = sys.modules.pop(mod, None)
        sys.modules[dep] = None  # type: ignore[assignment]  # None → ImportError on import


def _evict_dep_tree(dep: str) -> None:
    """Remove *dep* and all its submodules from ``sys.modules``."""
    for mod in list(sys.modules.keys()):
        if mod == dep or mod.startswith(dep + "."):
            sys.modules.pop(mod, None)


def _relink_parent_attributes(saved: dict[str, Any]) -> None:
    """Re-link parent package attributes after restoring ``sys.modules``.

    During a test, re-importing a module causes the import machinery to set
    a *new* module object as an attribute on the parent package.  Restoring
    ``sys.modules`` alone does not undo that ``setattr``, so the parent
    still references the tainted module.  Without this fix,
    ``unittest.mock.patch`` on dotted paths through the parent resolves to
    the stale (tainted) module object.
    """
    for mod, m in saved.items():
        if m is not None and "." in mod:
            parent_key, child_name = mod.rsplit(sep=".", maxsplit=1)
            parent_mod = sys.modules.get(parent_key)
            if parent_mod is not None:
                setattr(parent_mod, child_name, m)


def _restore_modules(saved: dict[str, Any], deps_to_mock: list[str], prefixes: list[str]) -> None:
    """Undo all ``sys.modules`` mutations made during the test.

    This does four things in order:
    1. Evicts any modules that were imported *during* the test (they hold
       references to mocked/missing dependencies and are not safe to reuse).
    2. Removes the ``None`` sentinels injected for mocked dependencies.
    3. Puts back the original module objects that were saved before the test.
    4. Re-links parent package attributes (see ``_relink_parent_attributes``).
    """
    _remove_modules_by_prefix(prefixes=prefixes)
    for dep in deps_to_mock:
        _evict_dep_tree(dep)
    for mod, m in saved.items():
        if m is not None:
            sys.modules[mod] = m
    _relink_parent_attributes(saved=saved)


@contextmanager
def mock_missing_deps(
    module_path: str,
    deps_to_mock: list[str],
    additional_modules: list[str] | None = None,
) -> Iterator[None]:
    """Context manager that simulates missing dependencies for import guard testing.

    **How it works, step by step:**

    1. Computes which ``sys.modules`` keys need to be evicted (the target
       module, its submodules, the corresponding private ``_``-prefixed
       implementation modules, and any *additional_modules*).
    2. Evicts those entries from ``sys.modules`` and saves the original
       module objects so they can be restored later.
    3. For each dependency in *deps_to_mock*, sets
       ``sys.modules[dep] = None``.  This is the standard CPython mechanism
       for making ``import dep`` raise ``ImportError``.
    4. **Yields** — test code runs here.  Any ``import`` of the target
       module will re-execute its module-level code and hit the mocked
       ``ImportError``, exercising the import guard.
    5. On exit, restores ``sys.modules`` to its original state so later
       tests are unaffected.

    Args:
        module_path: Dotted module path to evict (along with all submodules).
        deps_to_mock: Third-party packages to make unavailable by injecting
            ``None`` into ``sys.modules`` (e.g. ``["cv2"]``).
        additional_modules: Extra module paths to evict.  Needed when the
            target module imports from *other* project modules that also
            cache references to the mocked dependencies.

    Yields:
        None
    """
    saved: dict[str, Any] = {}
    prefixes = _get_module_prefixes(module_path)
    if additional_modules:
        for module in additional_modules:
            prefixes.extend(_get_module_prefixes(module))

    _clear_modules_by_prefix(prefixes=prefixes, saved=saved)
    _save_and_mock_deps(deps_to_mock=deps_to_mock, saved=saved)

    try:
        yield
    finally:
        _restore_modules(saved=saved, deps_to_mock=deps_to_mock, prefixes=prefixes)


class ImportGuardTestsMixin:
    """Mixin that provides standard import-guard tests.

    Subclasses only need to set the class-level attributes below.  The
    two inherited test methods (``test_import_error_message`` and
    ``test_attribute_error_for_unknown``) will then verify that:

    * Guarded classes raise a helpful ``ImportError`` when their
      dependency is unavailable.
    * Guarded classes are excluded from ``__all__`` when their
      dependency is unavailable.
    * Always-available classes remain in ``__all__``.
    * Accessing a truly non-existent attribute raises ``AttributeError``.

    Attributes:
        MODULE_PATH: Full dotted module path to test
            (e.g. ``"nrtk.impls.perturb_image.photometric.blur"``).
        DEPS_TO_MOCK: Third-party packages to simulate as missing
            (e.g. ``["cv2"]``).
        CLASSES: Class names that should be *unavailable* (raise
            ``ImportError``) when the dependencies are missing.
        ERROR_MATCH: Regex pattern to match against the ``ImportError``
            message.  Use ``{class_name}`` as a placeholder that will be
            formatted for each class.
        ALWAYS_AVAILABLE: Class names that should remain importable
            even when the mocked dependencies are absent.
        ADDITIONAL_MODULES: Extra module paths to evict from
            ``sys.modules`` (see ``mock_missing_deps``).
    """

    MODULE_PATH: str
    DEPS_TO_MOCK: list[str]
    CLASSES: list[str]
    ERROR_MATCH: str
    ALWAYS_AVAILABLE: list[str] = []
    ADDITIONAL_MODULES: list[str] = []

    def _get_module(self) -> ModuleType:
        """Import and return the module with mocked missing dependencies."""
        return import_module(self.MODULE_PATH)

    @pytest.mark.core
    def test_import_error_message(self) -> None:
        """Test that helpful ImportError is raised when dependency is unavailable."""
        with mock_missing_deps(
            module_path=self.MODULE_PATH,
            deps_to_mock=self.DEPS_TO_MOCK,
            additional_modules=self.ADDITIONAL_MODULES or None,
        ):
            module = self._get_module()

            # Always-available classes should be in __all__
            for class_name in self.ALWAYS_AVAILABLE:
                assert class_name in module.__all__

            # Guarded classes should NOT be in __all__ and should raise ImportError
            for class_name in self.CLASSES:
                assert class_name not in module.__all__
                with pytest.raises(ImportError, match=self.ERROR_MATCH.format(class_name=class_name)):
                    getattr(module, class_name)

    @pytest.mark.core
    def test_attribute_error_for_unknown(self) -> None:
        """Test that AttributeError is raised for unknown attributes."""
        with mock_missing_deps(
            module_path=self.MODULE_PATH,
            deps_to_mock=self.DEPS_TO_MOCK,
            additional_modules=self.ADDITIONAL_MODULES or None,
        ):
            module = self._get_module()
            with pytest.raises(
                AttributeError,
                match=r"NotARealClass",
            ):
                _ = module.NotARealClass
