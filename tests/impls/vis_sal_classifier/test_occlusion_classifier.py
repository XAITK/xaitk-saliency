from unittest import TestCase

import numpy as np

from xaitk_saliency.impls.vis_sal_classifier.occlusion_scoring import OcclusionScoring
from xaitk_saliency import ImageClassifierSaliencyMapGenerator


class TestOcclusionBasedScoring (TestCase):

    def test_init_(self) -> None:
        """
        Test if implementation is usable.
        """
        impl = OcclusionScoring()
        assert impl.is_usable() and isinstance(impl, ImageClassifierSaliencyMapGenerator)

# Below test case does pass but we should have more discussion
# regarding support for this

#     def test_multiple_images(self) -> None:
#         """
#         Test if implementation supports multiple images.
#         """
#         impl = OcclusionScoring()
#         # Three original images with two classes
#         image_confs_1_class_ = np.random.rand(3, 2)
#         # Three image confidences after perturbing with six masks and two classes
#         pertb_confs_1_class_ = np.random.rand(6, 3, 2)
#         # Six perturbation masks for and three images
#         mask_confs_1_class_ = np.random.randint(low=0, high=2, size=(3, 6, 10, 10), dtype='int')
#         # Final saliency map for two classes and 3 images of height and width 10px
#         sal = impl.generate(image_confs_1_class_, pertb_confs_1_class_, mask_confs_1_class_)
#         # Final saliency map for two classes and 3 images of height and width 10px
#         assert sal.shape == (2, 3, 10, 10)

    def test_1class_scores(self) -> None:
        """
        Test basic scoring with a single class for broadcasting sanity check.
        """
        impl = OcclusionScoring()
        # Three Pertrubation masks of height and width 10px for 1 class
        image_confs_1_class_ = np.random.rand(1)

        pertb_confs_1_class_ = np.random.rand(3, 1)
        mask_confs_1_class_ = np.random.randint(low=0, high=2, size=(3, 10, 10), dtype='int')

        sal = impl.generate(image_confs_1_class_, pertb_confs_1_class_, mask_confs_1_class_)
        assert sal.shape == (1, 10, 10)

    def test_standard_6class_scores(self) -> None:
        """
        Test basic scoring on known values.
        """
        impl = OcclusionScoring()
        # Three Pertrubation masks of size 4 x 6 for 6 classes
        image_confs_1_class_ = np.array([0.6])
        pertb_confs_1_class_ = np.array([[0.3], [0.6], [0.12], [0.18], [0.36], [0.42]])
        sal = impl.generate(image_confs_1_class_, pertb_confs_1_class_, EXPECTED_MASKS_4x6)
        assert sal.shape == (1, 4, 6) and np.sum(sal) == 1.08

    def test_nclass_scores(self) -> None:
        """
        Test scoring for n classes.
        """
        impl = OcclusionScoring()
        # Three Pertrubation masks of height and width 10px for 20 classes
        image_confs_1_class_ = np.random.rand(20)
        pertb_confs_1_class_ = np.random.rand(3, 20)
        mask_confs_1_class_ = np.random.randint(low=0, high=2, size=(3, 10, 10), dtype='int')
        sal = impl.generate(image_confs_1_class_, pertb_confs_1_class_, mask_confs_1_class_)
        assert sal.shape == (20, 10, 10)


# Common expected masks for 4x6 tests
EXPECTED_MASKS_4x6 = np.array([
    [[0, 0, 1, 1, 1, 1],
     [0, 0, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1]],
    [[1, 1, 0, 0, 1, 1],
     [1, 1, 0, 0, 1, 1],
     [1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1]],
    [[1, 1, 1, 1, 0, 0],
     [1, 1, 1, 1, 0, 0],
     [1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1]],
    [[1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1],
     [0, 0, 1, 1, 1, 1],
     [0, 0, 1, 1, 1, 1]],
    [[1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1],
     [1, 1, 0, 0, 1, 1],
     [1, 1, 0, 0, 1, 1]],
    [[1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 0, 0],
     [1, 1, 1, 1, 0, 0]],
], dtype=np.uint8)
