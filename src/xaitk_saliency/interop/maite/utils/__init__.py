"""Utils for generating saliency on detection datasets."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from xaitk_saliency._guard import Group, guard

if TYPE_CHECKING:
    from xaitk_saliency.interop.maite.utils._sal_on_dets import compute_sal_maps as compute_sal_maps
    from xaitk_saliency.interop.maite.utils._sal_on_dets import sal_on_dets as sal_on_dets

__getattr__: Callable[[str], Any]
__dir__: Callable[[], list[str]]
__all__: list[str]

__getattr__, __dir__, __all__ = guard(
    namespace=globals(),
    groups=[
        Group(
            symbols={
                "compute_sal_maps": "xaitk_saliency.interop.maite.utils._sal_on_dets",
                "sal_on_dets": "xaitk_saliency.interop.maite.utils._sal_on_dets",
            },
            extras=["maite"],
        ),
    ],
)
