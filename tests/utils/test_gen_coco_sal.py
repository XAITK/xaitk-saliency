import numpy as np
import os
import pytest

from xaitk_saliency.exceptions import MismatchedLabelsError

from tests import DATA_DIR
from . import TestDetector, TestPerturber, TestSalGenerator, MismatchedDetector, MismatchedNameDetector

try:
    import kwcoco  # type: ignore
    from xaitk_saliency.utils.gen_coco_sal import gen_coco_sal
    is_usable = True
except ImportError:
    is_usable = False

dets_file = os.path.join(DATA_DIR, 'test_dets.json')


@pytest.mark.skipif(not is_usable, reason="Extra 'xaitk-saliency[tools]' not installed.")
class TestGenCocoSal:

    def test_sal_gen(self) -> None:
        """
        Test saliency generation with dummy detections, detector, perturber,
        and saliency generator.
        """

        test_dets = kwcoco.CocoDataset(dets_file)
        test_detector = TestDetector()
        test_perturber = TestPerturber()
        test_sal_gen = TestSalGenerator()

        sal_maps = gen_coco_sal(test_dets, test_detector, test_perturber, test_sal_gen, verbose=True)

        maps_file = os.path.join(DATA_DIR, 'coco_sal_maps.npy')
        exp_sal_maps = np.load(maps_file, allow_pickle=True)[()]

        assert sal_maps.keys() == exp_sal_maps.keys()
        assert all([np.allclose(sal_maps[det_id], exp_sal_maps[det_id]) for det_id in sal_maps])

    def test_mismatched_labels(self) -> None:
        """
        Test that an exception is raised when test detections and detector have
        differing class labels.
        """

        test_dets = kwcoco.CocoDataset(dets_file)
        test_detector = MismatchedDetector()
        test_perturber = TestPerturber()
        test_sal_gen = TestSalGenerator()

        with pytest.raises(MismatchedLabelsError):
            gen_coco_sal(test_dets, test_detector, test_perturber, test_sal_gen)

    def test_mismatched_name_labels(self) -> None:
        """
        Test that an exception is raised when test detections and detector have
        differing string class labels.
        """
        test_dets = kwcoco.CocoDataset(dets_file)
        test_detector = MismatchedNameDetector()
        test_perturber = TestPerturber()
        test_sal_gen = TestSalGenerator()

        with pytest.raises(MismatchedLabelsError):
            gen_coco_sal(test_dets, test_detector, test_perturber, test_sal_gen)
