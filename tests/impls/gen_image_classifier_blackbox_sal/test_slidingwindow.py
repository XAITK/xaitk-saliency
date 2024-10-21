from collections.abc import Hashable, Iterator, Sequence

import numpy as np
from smqtk_classifier import ClassifyImage
from smqtk_classifier.interfaces.classification_element import CLASSIFICATION_DICT_T
from smqtk_classifier.interfaces.classify_image import IMAGE_ITER_T
from smqtk_core.configuration import configuration_test_helper

from xaitk_saliency.impls.gen_classifier_conf_sal.occlusion_scoring import OcclusionScoring
from xaitk_saliency.impls.gen_image_classifier_blackbox_sal.slidingwindow import SlidingWindowStack
from xaitk_saliency.impls.perturb_image.sliding_window import SlidingWindow


class TestSpecializationSlidingWindow:
    def test_configuration(self) -> None:
        """Test standard config things."""
        inst = SlidingWindowStack(
            window_size=(8, 9),
            stride=(19, 14),
            threads=99,
        )
        for inst_i in configuration_test_helper(inst):
            inst_p = inst_i._po._perturber
            inst_g = inst_i._po._generator
            assert isinstance(inst_p, SlidingWindow)
            assert isinstance(inst_g, OcclusionScoring)
            assert inst_p.window_size == (8, 9)
            assert inst_p.stride == (19, 14)
            assert inst_i._po._threads == 99

    def test_generation_rgb(self) -> None:
        """Test basic generation functionality with dummy image and blackbox"""

        class TestBlackBox(ClassifyImage):
            """Dummy blackbox that yields a constant result."""

            def get_labels(self) -> Sequence[Hashable]:
                return [0]

            def classify_images(self, img_iter: IMAGE_ITER_T) -> Iterator[CLASSIFICATION_DICT_T]:
                for _ in img_iter:
                    yield {0: 1.0}

            # Not implemented for stub
            get_config = None  # type: ignore

        test_image = np.full([32, 32, 3], fill_value=255, dtype=np.uint8)
        test_bb = TestBlackBox()

        inst = SlidingWindowStack((8, 8), (4, 4), threads=0)
        res = inst.generate(test_image, test_bb)
        # We expect this result to be composed of zeros because there is no
        # difference in classification performance across all sliding window
        # perturbations due to the fix return nature of the above stub class.
        exp_res = np.zeros(shape=(1, 32, 32))
        assert np.allclose(exp_res, res)

    def test_generation_gray(self) -> None:
        """Test basic generation functionality with dummy image and blackbox"""

        class TestBlackBox(ClassifyImage):
            """Dummy blackbox that yields a constant result."""

            def get_labels(self) -> Sequence[Hashable]:
                return [0]

            def classify_images(self, img_iter: IMAGE_ITER_T) -> Iterator[CLASSIFICATION_DICT_T]:
                for _ in img_iter:
                    yield {0: 1.0}

            # Not implemented for stub
            get_config = None  # type: ignore

        test_image = np.full([32, 32], fill_value=255, dtype=np.uint8)
        test_bb = TestBlackBox()

        inst = SlidingWindowStack((8, 8), (4, 4), threads=0)
        res = inst.generate(test_image, test_bb)
        # We expect this result to be composed of zeros because there is no
        # difference in classification performance across all sliding window
        # perturbations due to the fix return nature of the above stub class.
        exp_res = np.zeros(shape=(1, 32, 32))
        assert np.allclose(exp_res, res)

    def test_fill_prop(self) -> None:
        """
        Test that the `fill` property appropriately gets and sets the
        underlying `PerturbationOcclusion` instance fill instance attribute.
        """
        inst = SlidingWindowStack((8, 8), (4, 4), threads=0)
        assert inst.fill is None
        inst.fill = 42
        assert inst._po.fill == 42
