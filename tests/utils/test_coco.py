import os
import unittest.mock as mock
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image  # type: ignore

from tests import DATA_DIR
from xaitk_saliency.exceptions import KWCocoImportError
from xaitk_saliency.utils.coco import KWCocoUtils

if KWCocoUtils.is_usable():
    import kwcoco  # type: ignore


@pytest.mark.skipif(KWCocoUtils.is_usable(), reason="coco utils usable")
class TestParseCocoDsetNotUsable:
    @mock.patch.object(KWCocoUtils, "is_usable")
    def test_missing_deps(self, mock_is_usable: MagicMock) -> None:
        """Test that an exception is raised when required dependencies are not installed."""
        mock_is_usable.return_value = False
        assert not KWCocoUtils.is_usable()
        with pytest.raises(KWCocoImportError):
            KWCocoUtils()


@pytest.mark.skipif(not KWCocoUtils.is_usable(), reason=str(KWCocoImportError()))
class TestParseCocoDset:
    def test_dset_parse(self) -> None:
        """Test that a dummy detection file is parsed correctly."""
        dets_file = os.path.join(DATA_DIR, "test_dets.json")
        test_dset = kwcoco.CocoDataset(dets_file)  # type: ignore

        test_img_files = ["test_image1.png", "test_image2.png"]
        test_imgs = [np.array(Image.open(os.path.join(DATA_DIR, f))) for f in test_img_files]

        test_bboxes = [
            np.array([[10, 20, 40, 60], [12, 45, 62, 75], [30, 5, 77, 97]]),
            np.array([[50, 50, 82, 67], [68, 82, 79, 89]]),
        ]

        test_scores = [
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            np.array([[0.0, 1.0, 0.0], [0.2, 0.5, 0.3]]),
        ]

        inst = KWCocoUtils()
        for i, (img, bboxes, scores) in enumerate(inst.parse_coco_dset(test_dset)):  # type: ignore
            assert np.array_equal(img, test_imgs[i])
            assert np.array_equal(bboxes, test_bboxes[i])
            assert np.array_equal(scores, test_scores[i])
