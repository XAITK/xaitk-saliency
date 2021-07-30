from typing import Iterator, Sequence, Hashable

import numpy as np
from smqtk_classifier import ClassifyImage
from smqtk_classifier.interfaces.classification_element import CLASSIFICATION_DICT_T
from smqtk_classifier.interfaces.classify_image import IMAGE_ITER_T
from smqtk_core.configuration import configuration_test_helper

from xaitk_saliency.impls.gen_image_classifier_blackbox_sal.rise import (
    RISEGrid,
    RISEScoring,
    RISEStack,
)

from tests import DATA_DIR


class TestSpecializationRise:

    def test_configuration(self) -> None:
        """
        Test standard config things.
        """
        inst = RISEStack(
            n=444,
            s=33,
            p1=.22,
            seed=42,
            threads=99,
        )
        for inst_i in configuration_test_helper(inst):
            inst_p = inst_i._po.perturber
            inst_g = inst_i._po.generator
            assert isinstance(inst_p, RISEGrid)
            assert isinstance(inst_g, RISEScoring)
            assert inst_p.n == 444
            assert inst_p.s == 33
            assert inst_p.p1 == .22
            assert inst_p.seed == 42
            assert inst_p.threads == 99
            assert inst_g.p1 == .22
            assert inst_i.get_config()['debiased'] is True

    def test_configuration_not_debiased(self) -> None:
        """
        Test configuration when debiased is set to False.
        """
        inst = RISEStack(
            n=444,
            s=33,
            p1=.22,
            seed=42,
            threads=99,
            debiased=False,
        )
        for inst_i in configuration_test_helper(inst):
            inst_p = inst_i._po.perturber
            inst_g = inst_i._po.generator
            assert isinstance(inst_p, RISEGrid)
            assert isinstance(inst_g, RISEScoring)
            assert inst_p.p1 == .22
            assert inst_g.p1 == 0.0
            assert inst_i.get_config()['debiased'] is False

    def test_generation_rgb(self) -> None:
        """
        Test basic generation functionality with dummy image and blackbox
        """
        class TestBlackBox(ClassifyImage):
            """ Dummy blackbox that yields a constant result. """

            def get_labels(self) -> Sequence[Hashable]:
                return [0]

            def classify_images(self, img_iter: IMAGE_ITER_T) -> Iterator[CLASSIFICATION_DICT_T]:
                for _ in img_iter:
                    yield {0: 1.0}

            # Not implemented for stub
            get_config = None  # type: ignore

        test_image = np.full([32, 32, 3], fill_value=255, dtype=np.uint8)
        test_bb = TestBlackBox()

        # The heatmap result of this is merely the sum of RISE mask generation
        # normalized in the [-1,1] range as the generation stage does nothing
        # given the constant blackbox response.
        inst = RISEStack(5, 8, 0.5, seed=0)
        res = inst.generate(test_image, test_bb)

        exp_res = np.load(DATA_DIR / "exp_rise_stack_res.npy")
        assert np.allclose(exp_res, res)
