"""The one import guard every xaitk-saliency implementation module uses.

A module declares what it exposes and gets its :pep:`562` hooks back::

    __getattr__, __dir__, __all__ = guard(
        namespace=globals(),
        groups=[Group(symbols={"MAITEDetector": "xaitk_saliency...._model"}, extras=["maite"])],
    )

The contract -- how to declare a group, how to describe extras, and the rules
about pointing at leaf modules and never advertising a name ``__getattr__``
would reject -- lives in ``docs/design.rst``. Read that before changing anything
here; the constraints are not obvious from the code alone.
"""

from __future__ import annotations

import importlib
import itertools
from collections.abc import Callable, Mapping, Sequence
from types import ModuleType
from typing import Any, NamedTuple

# An extras requirement. The outer sequence is AND, a nested sequence is OR:
#   ["maite"]                       -> maite
#   ["maite", "tools"]              -> maite AND tools
#   [("a", "b")]                    -> a OR b
#   ["c", ("a", "b")]               -> both, with a choice for the second
Extras = Sequence["str | Sequence[str]"]


class Group(NamedTuple):
    """Symbols that share one requirement.

    Attributes:
        symbols: Public name -> the leaf module that defines it.
        extras: The extras required, or empty for none. See :data:`Extras`.
    """

    symbols: Mapping[str, str]
    extras: Extras = ()


def _phrase(requirement: str | Sequence[str]) -> str:
    """Render one AND-term. An OR-term is always bracketed, so `a` and (`b` or `c`) group."""
    if isinstance(requirement, str):
        return f"`{requirement}`"
    return f"(`{'` or `'.join(requirement)}`)"


def _requirement(extras: Extras) -> str:
    """Describe *extras* in prose, e.g. "the `maite` and `tools` extras"."""
    terms = [_phrase(requirement) for requirement in extras]
    return f"the {' and '.join(terms)} {'extras' if len(extras) > 1 else 'extra'}"


def _install_hint(extras: Extras) -> str:
    """Spell out the install command(s), one per way of satisfying every OR-term."""
    choices = [(term,) if isinstance(term, str) else tuple(term) for term in extras]
    return " or ".join(f"`pip install xaitk-saliency[{','.join(combo)}]`" for combo in itertools.product(*choices))


def extra_error(*, name: str, extras: Extras, cause: ImportError | None) -> ImportError:
    """Build the standard "you are missing an extra" error for *name*.

    Args:
        name: The symbol the caller asked for.
        extras: What it needs. See :data:`Extras`.
        cause: The import failure that was actually observed, if there was one.

    Returns:
        The error to raise. Callers raise it themselves so the ``from cause``
        chaining decision stays where the traceback is honest.
    """
    msg = f"{name} requires {_requirement(extras)}. Install with: {_install_hint(extras)}"
    if cause is not None:
        msg += (
            "\n\nIf the extra is already installed, the following upstream error may be the cause:"
            f"\n  {type(cause).__name__}: {cause}"
        )
    return ImportError(msg)


class _Guard:
    """Backs one module's ``__getattr__``/``__dir__``. Built by :func:`guard`."""

    def __init__(self, *, namespace: dict[str, Any], groups: Sequence[Group], submodules: Sequence[str]) -> None:
        self._namespace = namespace
        self._module = namespace["__name__"]
        self._groups = list(groups)
        self._submodules = list(submodules)
        self.all: list[str] = list(submodules)

        # Map every declared name to the group that owns it, so __getattr__ is one lookup.
        self._owner: dict[str, int] = {}
        for index, group in enumerate(self._groups):
            self._owner.update(dict.fromkeys(group.symbols, index))

        # Resolve every group now; installed extras do not change at runtime.
        self._causes: list[ImportError | None] = [self._load(group=group) for group in self._groups]

    def _load(self, *, group: Group) -> ImportError | None:
        """Bind a group's symbols, or report the failure that says its extra is missing."""
        try:
            for name, source in group.symbols.items():
                self._bind(name=name, source=source)
        except ImportError as ex:
            if not group.extras:
                raise  # nothing optional here, so a failure is a bug rather than a missing extra
            return ex
        self.all.extend(group.symbols)  # only advertise what actually loaded
        return None

    def _bind(self, *, name: str, source: str) -> object:
        """Import *name* from its leaf module, re-home it, and cache it in the namespace."""
        obj = getattr(importlib.import_module(source), name)
        # Re-home so smqtk records the public path in serialized configs, not the private leaf.
        # This mutates the class globally, so exactly one group may claim a given symbol.
        obj.__module__ = self._module
        self._namespace[name] = obj  # PEP 562: the hook stops firing for a name once it is bound
        return obj

    def _submodule(self, name: str) -> ModuleType:
        """Import a declared submodule on first access, or reject an unknown name."""
        if name not in self._submodules:
            raise AttributeError(f"module {self._module!r} has no attribute {name!r}")
        module = importlib.import_module(f"{self._module}.{name}")
        self._namespace[name] = module
        return module

    def getattr(self, name: str) -> object:
        """Serve as the module's :pep:`562` ``__getattr__``."""
        index = self._owner.get(name)
        if index is None:
            return self._submodule(name)
        # Symbols load eagerly, so reaching the hook means the extra is missing.
        group = self._groups[index]
        raise extra_error(name=name, extras=group.extras, cause=self._causes[index])

    def dir(self) -> list[str]:
        """Serve as the module's :pep:`562` ``__dir__``."""
        return sorted(set(self.all))


def guard(
    *,
    namespace: dict[str, Any],
    groups: Sequence[Group] = (),
    submodules: Sequence[str] = (),
) -> tuple[Callable[[str], Any], Callable[[], list[str]], list[str]]:
    """Install xaitk-saliency's import guard for the calling module.

    Args:
        namespace: The module's ``globals()``. Read for ``__name__``, written to
            cache resolved symbols so the hook stops firing for them.
        groups: What the module exposes, one :class:`Group` per requirement.
        submodules: Submodules to expose lazily, imported on first access.

    Returns:
        ``(__getattr__, __dir__, __all__)``, to be unpacked into those names.
        ``__all__`` holds the submodules plus every symbol that resolved.
    """
    module_guard = _Guard(namespace=namespace, groups=groups, submodules=submodules)
    return module_guard.getattr, module_guard.dir, module_guard.all
