"""Test import guard behavior for the image_classification model module."""

import pytest

from tests._utils.import_guard_tests_mixin import ImportGuardTestsMixin


@pytest.mark.maite
class TestModelMaiteImportGuard(ImportGuardTestsMixin):
    MODULE_PATH = "xaitk_saliency.interop.maite.image_classification"
    DEPS_TO_MOCK = ["maite"]
    CLASSES = ["MAITEImageClassifier"]
    ERROR_MATCH = r"{class_name} requires the `maite` extra\. Install with: `pip install xaitk-saliency\[maite\]`"


@pytest.mark.maite
@pytest.mark.usefixtures("require_marker")
def test_model_maite_public_imports() -> None:
    """Canary test: FAIL if maite marker is used but MAITEImageClassifier can't be imported."""
    try:
        from xaitk_saliency.interop.maite.image_classification import MAITEImageClassifier

        del MAITEImageClassifier
    except ImportError as e:
        pytest.fail(f"Running with maite marker but classes not importable: {e}. Ensure maite extra is installed.")
