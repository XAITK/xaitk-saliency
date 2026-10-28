"""Public module fronting ``DatamaiteUtils``. Guards the ``datamaite`` optional dependency."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from xaitk_saliency._guard import Group, guard

if TYPE_CHECKING:
    from xaitk_saliency.utils._datamaite import DatamaiteUtils as DatamaiteUtils

__getattr__: Callable[[str], Any]
__dir__: Callable[[], list[str]]
__all__: list[str]

__getattr__, __dir__, __all__ = guard(
    namespace=globals(),
    groups=[
        Group(
            symbols={"DatamaiteUtils": "xaitk_saliency.utils._datamaite"},
            extras=["tools"],
        ),
    ],
)
