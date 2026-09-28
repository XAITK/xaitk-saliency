"""Test import guard behavior for the utils sal_on_dets module."""

import pytest

from tests._utils.import_guard_tests_mixin import ImportGuardTestsMixin


@pytest.mark.maite
class TestSalOnDetsMaiteImportGuard(ImportGuardTestsMixin):
    MODULE_PATH = "xaitk_saliency.interop.maite.utils"
    DEPS_TO_MOCK = ["maite"]
    CLASSES = ["compute_sal_maps", "sal_on_dets"]
    ERROR_MATCH = r"{class_name} requires the `maite` extra\. Install with: `pip install xaitk-saliency\[maite\]`"


@pytest.mark.maite
@pytest.mark.usefixtures("require_marker")
def test_sal_on_dets_maite_public_imports() -> None:
    """Canary test: FAIL if maite marker is used but classes can't be imported."""
    try:
        from xaitk_saliency.interop.maite.utils import compute_sal_maps, sal_on_dets

        del compute_sal_maps
        del sal_on_dets
    except ImportError as e:
        pytest.fail(f"Running with maite marker but classes not importable: {e}. Ensure maite extra is installed.")
