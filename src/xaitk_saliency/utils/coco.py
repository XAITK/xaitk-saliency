"""Public module fronting ``KWCocoUtils``. Guards the ``kwcoco`` optional dependency."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from xaitk_saliency._guard import Group, guard

if TYPE_CHECKING:
    from xaitk_saliency.utils._coco import KWCocoUtils as KWCocoUtils

__getattr__: Callable[[str], Any]
__dir__: Callable[[], list[str]]
__all__: list[str]

__getattr__, __dir__, __all__ = guard(
    namespace=globals(),
    groups=[
        Group(
            symbols={"KWCocoUtils": "xaitk_saliency.utils._coco"},
            extras=["tools"],
        ),
    ],
)
