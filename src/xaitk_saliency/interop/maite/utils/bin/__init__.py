"""Utils for generating saliency on coco datasets.

This module provides import guards for optional dependencies:
- sal_on_coco_dets: requires ``maite`` and ``tools`` extras
"""

from xaitk_saliency.exceptions import MaiteImportError, ToolsImportError

_MAITE_NAMES = ["sal_on_coco_dets"]

__all__: list[str] = []

_import_error: ImportError | None = None

try:
    from xaitk_saliency.interop.maite.utils.bin._sal_on_coco_dets import compute_sal_maps as compute_sal_maps
    from xaitk_saliency.interop.maite.utils.bin._sal_on_coco_dets import sal_on_coco_dets as sal_on_coco_dets

    __all__ += _MAITE_NAMES  # pyright: ignore[reportUnsupportedDunderAll]
except ImportError as _ex:
    _import_error = _ex


def __getattr__(name: str) -> None:
    if name in _MAITE_NAMES:
        missing_module = getattr(_import_error, "name", None) or ""
        if missing_module == "maite" or missing_module.startswith("maite."):
            raise MaiteImportError(name, import_error=_import_error)
        if any(missing_module == package for package in ("click", "kwcoco")):
            raise ToolsImportError from _import_error
        if _import_error is not None:
            raise _import_error
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
