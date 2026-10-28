"""Utils for generating saliency on object detection datasets loaded via ``datamaite``."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from xaitk_saliency._guard import Group, guard

if TYPE_CHECKING:
    from xaitk_saliency.interop.maite.utils.bin._sal_on_od_dets import sal_on_od_dets as sal_on_od_dets

__getattr__: Callable[[str], Any]
__dir__: Callable[[], list[str]]
__all__: list[str]

__getattr__, __dir__, __all__ = guard(
    namespace=globals(),
    groups=[
        Group(
            symbols={
                "sal_on_od_dets": "xaitk_saliency.interop.maite.utils.bin._sal_on_od_dets",
            },
            extras=["maite", "tools"],
        ),
    ],
)
