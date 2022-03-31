import numpy as np
import os
import pytest
from PIL import Image  # type: ignore
import builtins
import sys
from typing import Any

from tests import DATA_DIR

try:
    import kwcoco  # type: ignore
    from xaitk_saliency.utils.coco import parse_coco_dset
    is_usable = True
except ImportError:
    is_usable = False


class TestGenCocoSalNotUsable:

    def test_error(self) -> None:
        """
        Test that proper error is raised when required dependencies are not
        installed.
        """

        if is_usable:
            real_import = builtins.__import__

            # mock import function that acts as if kwcoco is not installed
            def mock_import(name: str, *args: Any, **kw: Any) -> None:
                if name == 'kwcoco':
                    raise ModuleNotFoundError
                return real_import(name, *args, **kw)

            # monkeypatch import function
            builtins.__import__ = mock_import

            del sys.modules['xaitk_saliency.utils.coco']

        with pytest.raises(ImportError):
            from xaitk_saliency.utils.coco import parse_coco_dset  # noqa: F401


@pytest.mark.skipif(not is_usable, reason="Extra 'xaitk-saliency[tools]' not installed.")
class TestParseCocoDset:

    def test_dset_parse(self) -> None:
        """
        Test that a dummy detection file is parsed correctly.
        """
        dets_file = os.path.join(DATA_DIR, 'test_dets.json')
        test_dset = kwcoco.CocoDataset(dets_file)

        test_img_files = ['test_image1.png', 'test_image2.png']
        test_imgs = [np.array(Image.open(os.path.join(DATA_DIR, f))) for f in test_img_files]

        test_bboxes = [
            np.array([
                [10, 20, 40, 60],
                [12, 45, 62, 75],
                [30, 5, 77, 97]
            ]),
            np.array([
                [50, 50, 82, 67],
                [68, 82, 79, 89]
            ])
        ]

        test_scores = [
            np.array([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ]),
            np.array([
                [0.0, 1.0, 0.0],
                [0.2, 0.5, 0.3]
            ])
        ]

        for i, (img, bboxes, scores) in enumerate(parse_coco_dset(test_dset)):
            assert np.array_equal(img, test_imgs[i])
            assert np.array_equal(bboxes, test_bboxes[i])
            assert np.array_equal(scores, test_scores[i])
