import os

import numpy as np
import pytest
from PIL import Image  # type: ignore

import xaitk_saliency.utils.coco
from tests import DATA_DIR

try:
    import kwcoco  # type: ignore

    from xaitk_saliency.utils.coco import parse_coco_dset
except ImportError:
    # Won't use above imports when not importable
    pass


@pytest.mark.skipif(xaitk_saliency.utils.coco.is_usable, reason="coco utils usable")
class TestParseCocoDsetNotUsable:
    def test_func_not_exposed(self) -> None:
        """
        Test that if we try to import the `parse_coco_dset` function when
        kwcoco is not installed, we get an import error, due to the function
        being hidden behind and if-else conditioned on kwcoco's successful
        import.
        """
        # no error
        import xaitk_saliency.utils.coco  # noqa: F401

        # now the error
        with pytest.raises(ImportError):
            from xaitk_saliency.utils.coco import parse_coco_dset  # noqa: F401


@pytest.mark.skipif(not xaitk_saliency.utils.coco.is_usable, reason="Extra 'xaitk-saliency[tools]' not installed.")
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

        for i, (img, bboxes, scores) in enumerate(parse_coco_dset(test_dset)):  # type: ignore
            assert np.array_equal(img, test_imgs[i])
            assert np.array_equal(bboxes, test_bboxes[i])
            assert np.array_equal(scores, test_scores[i])
