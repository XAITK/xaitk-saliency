"""Public API for the ``sal_on_coco_dets`` CLI. Guards the ``click`` optional dependency.

This module provides import guards for optional dependencies:
- sal_on_coco_dets: requires ``click`` (the ``tools`` extra)
"""

from xaitk_saliency.exceptions import ToolsImportError

_TOOLS_NAMES = ["sal_on_coco_dets"]

__all__: list[str] = []

_import_error: ImportError | None = None

try:
    from xaitk_saliency.utils.bin._sal_on_coco_dets import sal_on_coco_dets as sal_on_coco_dets

    __all__ += _TOOLS_NAMES  # pyright: ignore[reportUnsupportedDunderAll]
except ImportError as _ex:
    _import_error = _ex


def __getattr__(name: str) -> None:
    if name in _TOOLS_NAMES:
        missing_module = getattr(_import_error, "name", None) or ""
        if any(missing_module == package for package in ("click", "kwcoco")):
            raise ToolsImportError from _import_error
        if _import_error is not None:
            raise _import_error
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
