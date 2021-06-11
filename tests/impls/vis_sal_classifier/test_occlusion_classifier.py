from unittest import TestCase

import numpy as np
import os

from xaitk_saliency.impls.vis_sal_classifier.occlusion_scoring import OcclusionScoring
from xaitk_saliency import ImageClassifierSaliencyMapGenerator
from tests import DATA_DIR, EXPECTED_MASKS_4x6


class TestOcclusionBasedScoring (TestCase):

    def test_init_(self) -> None:
        """
        Test if implementation is usable.
        """
        impl = OcclusionScoring()
        assert impl.is_usable() and isinstance(impl, ImageClassifierSaliencyMapGenerator)

    def test_1class_scores(self) -> None:
        """
        Test basic scoring with a single class for broadcasting sanity check.
        """
        impl = OcclusionScoring()
        np.random.seed(2)
        # Three Pertrubation masks of height and width 10px for 1 class
        image_confs_1_class_ = np.random.rand(1)
        pertb_confs_1_class_ = np.random.rand(3, 1)
        mask_confs_1_class_ = np.random.randint(low=0, high=2, size=(3, 10, 10), dtype='int')

        sal = impl.generate(image_confs_1_class_, pertb_confs_1_class_, mask_confs_1_class_)
        assert sal.shape == (1, 10, 10)

    def test_standard_1class_scores(self) -> None:
        """
        Test basic scoring on known values and non-square masks.
        """
        impl = OcclusionScoring()
        # Three Pertrubation masks of size 4 x 6 for 1 class
        image_confs_1_class_ = np.array([0.6])
        pertb_confs_1_class_ = np.array([[0.3], [0.65], [0.12], [0.18], [0.36], [0.42]])
        sal = impl.generate(image_confs_1_class_, pertb_confs_1_class_, EXPECTED_MASKS_4x6)
        standard_sal = np.load(os.path.join(DATA_DIR, 'OccScorSal.npy'))
        assert sal.shape == (1, 4, 6) and round(np.sum(sal)) == 14 and np.array_equal(standard_sal, sal)

    def test_20class_scores(self) -> None:
        """
        Test scoring for 20 classes.
        """
        impl = OcclusionScoring()
        np.random.seed(2)
        # Three Pertrubation masks of height and width 10px for 20 classes
        image_confs_1_class_ = np.random.rand(20)
        pertb_confs_1_class_ = np.random.rand(3, 20)
        mask_confs_1_class_ = np.random.randint(low=0, high=2, size=(3, 10, 10), dtype='int')
        sal = impl.generate(image_confs_1_class_, pertb_confs_1_class_, mask_confs_1_class_)
        assert sal.shape == (20, 10, 10)
