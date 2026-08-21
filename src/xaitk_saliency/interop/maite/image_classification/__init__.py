"""Implementations to use XAITK for image classifcation in a maite compliant way.

This module provides import guards for optional dependencies:
- MAITEImageClassifier: requires ``maite`` extra
"""

from xaitk_saliency.exceptions import MaiteImportError

_MAITE_CLASSES = ["MAITEImageClassifier"]

__all__: list[str] = []

_import_error: ImportError | None = None

try:
    from xaitk_saliency.interop.maite.image_classification._model import MAITEImageClassifier as MAITEImageClassifier

    __all__ += _MAITE_CLASSES  # pyright: ignore[reportUnsupportedDunderAll]
    # Re-home so Sphinx documents the public path, not the private leaf.
    for _cls_name in _MAITE_CLASSES:
        globals()[_cls_name].__module__ = __name__
except ImportError as _ex:
    _import_error = _ex


def __getattr__(name: str) -> None:
    if name in _MAITE_CLASSES:
        raise MaiteImportError(name, import_error=_import_error)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
