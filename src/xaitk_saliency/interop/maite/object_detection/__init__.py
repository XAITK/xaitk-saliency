"""MAITE compliant dataset and model objects for using XAITK for object detection."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from xaitk_saliency._guard import Group, guard

if TYPE_CHECKING:
    from xaitk_saliency.interop.maite.object_detection._coco_maite_object_detection_dataset import (
        COCOMAITEObjectDetectionDataset as COCOMAITEObjectDetectionDataset,
    )
    from xaitk_saliency.interop.maite.object_detection._maite_object_detection_dataset import (
        MAITEDetectionTarget as MAITEDetectionTarget,
    )
    from xaitk_saliency.interop.maite.object_detection._maite_object_detection_dataset import (
        MAITEObjectDetectionDataset as MAITEObjectDetectionDataset,
    )
    from xaitk_saliency.interop.maite.object_detection._model import MAITEDetector as MAITEDetector

_OBJECT_DETECTION = "xaitk_saliency.interop.maite.object_detection"

__getattr__: Callable[[str], Any]
__dir__: Callable[[], list[str]]
__all__: list[str]

__getattr__, __dir__, __all__ = guard(
    namespace=globals(),
    groups=[
        Group(
            symbols={
                "MAITEDetectionTarget": f"{_OBJECT_DETECTION}._maite_object_detection_dataset",
                "MAITEObjectDetectionDataset": f"{_OBJECT_DETECTION}._maite_object_detection_dataset",
                "MAITEDetector": f"{_OBJECT_DETECTION}._model",
            },
            extras=["maite"],
        ),
        Group(
            symbols={
                "COCOMAITEObjectDetectionDataset": f"{_OBJECT_DETECTION}._coco_maite_object_detection_dataset",
            },
            extras=["maite", "tools"],
        ),
    ],
)
