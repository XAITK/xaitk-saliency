"""Implementations to use XAITK for image classifcation in a maite compliant way."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from xaitk_saliency._guard import Group, guard

if TYPE_CHECKING:
    from xaitk_saliency.interop.maite.image_classification._model import MAITEImageClassifier as MAITEImageClassifier

__getattr__: Callable[[str], Any]
__dir__: Callable[[], list[str]]
__all__: list[str]

__getattr__, __dir__, __all__ = guard(
    namespace=globals(),
    groups=[
        Group(
            symbols={
                "MAITEImageClassifier": "xaitk_saliency.interop.maite.image_classification._model",
            },
            extras=["maite"],
        ),
    ],
)
