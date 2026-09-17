"""Utils for generating saliency on detection datasets.

This module provides import guards for optional dependencies:
- compute_sal_maps, sal_on_dets: require ``maite`` extra
"""

from xaitk_saliency.exceptions import MaiteImportError

_MAITE_NAMES = ["compute_sal_maps", "sal_on_dets"]

__all__: list[str] = []

_import_error: ImportError | None = None

try:
    from xaitk_saliency.interop.maite.utils._sal_on_dets import compute_sal_maps as compute_sal_maps
    from xaitk_saliency.interop.maite.utils._sal_on_dets import sal_on_dets as sal_on_dets

    __all__ += _MAITE_NAMES  # pyright: ignore[reportUnsupportedDunderAll]
except ImportError as _ex:
    _import_error = _ex


def __getattr__(name: str) -> None:
    if name in _MAITE_NAMES:
        raise MaiteImportError(name, import_error=_import_error)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
