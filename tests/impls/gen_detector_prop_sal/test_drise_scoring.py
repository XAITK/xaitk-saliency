from unittest import TestCase

import numpy as np
import os

from xaitk_saliency.impls.gen_detector_prop_sal.drise_scoring import DetectorRISE

from xaitk_saliency import GenerateDetectorProposalSaliency
from smqtk_core.configuration import configuration_test_helper
from tests import DATA_DIR, EXPECTED_MASKS_4x6


class TestSimilarityScoring (TestCase):

    def test_init_(self) -> None:
        """
        Test if implementation is usable.
        """
        impl = DetectorRISE()
        assert impl.is_usable() and isinstance(impl, GenerateDetectorProposalSaliency)

    def test_default_param(self) -> None:
        """
        Test default construction.
        """
        impl = DetectorRISE()
        assert impl.proximity_metric == 'cosine'

    def test_get_config(self) -> None:
        """
        Test expected configuation behavior.
        """
        impl = DetectorRISE('euclidean')
        for i in configuration_test_helper(impl):
            assert i.proximity_metric == 'euclidean'

    def test_metric_args(self) -> None:
        """
        Test non-default metric type.
        """
        impl = DetectorRISE('hamming')
        assert impl.proximity_metric == 'hamming'

    def test_shape_sanity(self) -> None:
        """
        Test basic scoring with a single feature for broadcasting sanity check.
        """
        impl = DetectorRISE()
        np.random.seed(2)
        image1_dets = np.random.rand(2, (7))
        pertb_dets = np.random.rand(10, 2, (7))
        pertb_mask = np.random.randint(low=0, high=2, size=(10, 15, 25), dtype='int')
        sal = impl.generate(image1_dets, pertb_dets, pertb_mask)
        assert sal.shape == (2, 15, 25)

    def test_standard_detection(self) -> None:
        """
        Test basic scoring on known values and non-square masks.
        """

        impl = DetectorRISE()
        image1_dets = np.array([[1, 1, 4, 3, 0, 1, 0.89]])
        pertb_dets = np.array([[[1, 2, 6, 6, 0.3, 1, 0.995]],
                               [[0, 1, 2, 2, 0.2, 2, 0.03]],
                               [[1, 0, 2, 2, 0.45, 1, 0.81]],
                               [[1, 1, 6, 6, 0.5, 1, 0.625]],
                               [[0, 2, 3, 5, 0.03, 1, 0.56]],
                               [[1, 2, 6, 3, 0.01, 1, 0.07]], ])
        sal = impl.generate(image1_dets, pertb_dets, EXPECTED_MASKS_4x6)
        standard_sal = np.load(os.path.join(DATA_DIR, 'drisesal.npy'))
        assert sal.shape == (1, 4, 6)
        assert np.allclose(standard_sal, sal)
