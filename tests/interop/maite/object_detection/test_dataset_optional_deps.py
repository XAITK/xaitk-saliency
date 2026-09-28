"""Test import guard behavior for the object_detection dataset module."""

import pytest

from tests._utils.import_guard_tests_mixin import ImportGuardTestsMixin


@pytest.mark.maite
class TestDatasetMaiteImportGuard(ImportGuardTestsMixin):
    MODULE_PATH = "xaitk_saliency.interop.maite.object_detection"
    DEPS_TO_MOCK = ["maite"]
    CLASSES = [
        "MAITEDetectionTarget",
        "MAITEObjectDetectionDataset",
        "COCOMAITEObjectDetectionDataset",
    ]
    ERROR_MATCH = r"{class_name} requires the `maite` extra\. Install with: `pip install xaitk-saliency\[maite\]`"


@pytest.mark.maite
@pytest.mark.usefixtures("require_marker")
def test_dataset_maite_public_imports() -> None:
    """Canary test: FAIL if maite marker is used but the dataset classes can't be imported."""
    try:
        from xaitk_saliency.interop.maite.object_detection import (
            COCOMAITEObjectDetectionDataset,
            MAITEDetectionTarget,
            MAITEObjectDetectionDataset,
        )

        del MAITEDetectionTarget
        del MAITEObjectDetectionDataset
        del COCOMAITEObjectDetectionDataset
    except ImportError as e:
        pytest.fail(f"Running with maite marker but classes not importable: {e}. Ensure maite extra is installed.")
