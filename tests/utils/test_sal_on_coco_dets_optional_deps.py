"""Test import guard behavior for the sal_on_coco_dets CLI module."""

import pytest

from tests._utils.import_guard_tests_mixin import ImportGuardTestsMixin

TOOLS_ERROR_MATCH = r"This feature requires additional dependencies\. Please install via `xaitk-saliency\[tools\]`\."


@pytest.mark.tools
class TestSalOnCocoDetsClickImportGuard(ImportGuardTestsMixin):
    MODULE_PATH = "xaitk_saliency.utils.bin"
    DEPS_TO_MOCK = ["click"]
    CLASSES = ["sal_on_coco_dets"]
    ERROR_MATCH = TOOLS_ERROR_MATCH


@pytest.mark.tools
class TestSalOnCocoDetsKWCocoImportGuard(ImportGuardTestsMixin):
    MODULE_PATH = "xaitk_saliency.utils.bin"
    DEPS_TO_MOCK = ["kwcoco"]
    CLASSES = ["sal_on_coco_dets"]
    ERROR_MATCH = TOOLS_ERROR_MATCH


@pytest.mark.tools
@pytest.mark.usefixtures("require_marker")
def test_sal_on_coco_dets_tools_public_imports() -> None:
    """Canary test: FAIL if tools marker is used but sal_on_coco_dets can't be imported."""
    try:
        from xaitk_saliency.utils.bin import sal_on_coco_dets

        del sal_on_coco_dets
    except ImportError as e:
        pytest.fail(f"Running with tools marker but classes not importable: {e}. Ensure tools extra is installed.")
