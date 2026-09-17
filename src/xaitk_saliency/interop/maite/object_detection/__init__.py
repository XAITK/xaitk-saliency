"""MAITE compliant dataset and model objects for using XAITK for object detection.

This module provides import guards for optional dependencies:
- MAITEDetectionTarget, MAITEObjectDetectionDataset, MAITEDetector,
  COCOMAITEObjectDetectionDataset: require ``maite`` extra
"""

from xaitk_saliency.exceptions import MaiteImportError

_MAITE_CLASSES = [
    "COCOMAITEObjectDetectionDataset",
    "MAITEDetectionTarget",
    "MAITEObjectDetectionDataset",
    "MAITEDetector",
]

__all__: list[str] = []

_import_error: ImportError | None = None

try:
    from xaitk_saliency.interop.maite.object_detection._dataset import (
        COCOMAITEObjectDetectionDataset as COCOMAITEObjectDetectionDataset,
    )
    from xaitk_saliency.interop.maite.object_detection._dataset import (
        MAITEDetectionTarget as MAITEDetectionTarget,
    )
    from xaitk_saliency.interop.maite.object_detection._dataset import (
        MAITEObjectDetectionDataset as MAITEObjectDetectionDataset,
    )
    from xaitk_saliency.interop.maite.object_detection._model import MAITEDetector as MAITEDetector

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
