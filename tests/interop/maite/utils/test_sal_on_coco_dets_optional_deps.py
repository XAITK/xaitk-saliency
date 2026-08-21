"""Test import guard behavior for the MAITE sal_on_coco_dets CLI module."""

import pytest

from tests._utils.import_guard_tests_mixin import ImportGuardTestsMixin

TOOLS_ERROR_MATCH = r"This feature requires additional dependencies\. Please install via `xaitk-saliency\[tools\]`\."


@pytest.mark.maite
@pytest.mark.tools
class TestSalOnCocoDetsMaiteImportGuard(ImportGuardTestsMixin):
    MODULE_PATH = "xaitk_saliency.interop.maite.utils.bin"
    DEPS_TO_MOCK = ["maite"]
    CLASSES = ["sal_on_coco_dets"]
    ERROR_MATCH = r"{class_name} requires the `maite` extra\. Install with: `pip install xaitk-saliency\[maite\]`"
    ADDITIONAL_MODULES = [
        "xaitk_saliency.interop.maite.utils",
        "xaitk_saliency.interop.maite.object_detection",
    ]


@pytest.mark.maite
@pytest.mark.tools
class TestSalOnCocoDetsClickImportGuard(ImportGuardTestsMixin):
    MODULE_PATH = "xaitk_saliency.interop.maite.utils.bin"
    DEPS_TO_MOCK = ["click"]
    CLASSES = ["sal_on_coco_dets"]
    ERROR_MATCH = TOOLS_ERROR_MATCH
    ADDITIONAL_MODULES = [
        "xaitk_saliency.interop.maite.utils",
        "xaitk_saliency.interop.maite.object_detection",
    ]


@pytest.mark.maite
@pytest.mark.tools
class TestSalOnCocoDetsKWCocoImportGuard(ImportGuardTestsMixin):
    MODULE_PATH = "xaitk_saliency.interop.maite.utils.bin"
    DEPS_TO_MOCK = ["kwcoco"]
    CLASSES = ["sal_on_coco_dets"]
    ERROR_MATCH = TOOLS_ERROR_MATCH
    ADDITIONAL_MODULES = [
        "xaitk_saliency.interop.maite.utils",
        "xaitk_saliency.interop.maite.object_detection",
    ]


@pytest.mark.maite
@pytest.mark.tools
@pytest.mark.usefixtures("require_marker")
def test_sal_on_coco_dets_maite_public_imports() -> None:
    """Canary test: FAIL if maite marker is used but classes can't be imported."""
    try:
        from xaitk_saliency.interop.maite.utils.bin import (
            compute_sal_maps,
            sal_on_coco_dets,
        )

        del compute_sal_maps
        del sal_on_coco_dets
    except ImportError as e:
        pytest.fail(f"Running with maite marker but classes not importable: {e}. Ensure maite extra is installed.")
