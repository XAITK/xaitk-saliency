from collections.abc import Hashable, Iterator, Sequence

import numpy as np
import pytest
from smqtk_classifier.interfaces.classification_element import CLASSIFICATION_DICT_T
from smqtk_classifier.interfaces.classify_image import IMAGE_ITER_T, ClassifyImage
from smqtk_core.configuration import configuration_test_helper
from syrupy.assertion import SnapshotAssertion

from tests.test_utils import CustomFloatSnapshotExtension
from xaitk_saliency.impls.gen_image_classifier_blackbox_sal.mc_rise import (
    MCRISEGrid,
    MCRISEScoring,
    MCRISEStack,
)


@pytest.fixture
def snapshot_custom(snapshot: SnapshotAssertion) -> SnapshotAssertion:
    return snapshot.use_extension(lambda: CustomFloatSnapshotExtension())  # type: ignore


class TestMCRise:
    def test_configuration(self) -> None:
        """Test standard config things."""
        fill_colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]]
        inst = MCRISEStack(
            n=444,
            s=33,
            p1=0.22,
            fill_colors=fill_colors,
            seed=42,
            threads=99,
        )
        for inst_i in configuration_test_helper(inst):
            inst_p = inst_i._perturber
            inst_g = inst_i._generator
            assert isinstance(inst_p, MCRISEGrid)
            assert isinstance(inst_g, MCRISEScoring)
            assert inst_p.n == 444
            assert inst_p.s == 33
            assert np.allclose(inst_p.p1, 0.22)
            assert inst_p.k == len(fill_colors)
            assert inst_p.seed == 42
            assert inst_p.threads == 99
            assert inst_g.k == len(fill_colors)
            assert np.allclose(inst_g.p1, 0.22)
            assert inst_i._threads == 99
            assert inst_i._fill_colors == fill_colors

    def test_generation_rgb(self, snapshot_custom: SnapshotAssertion) -> None:
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

        # The heatmap result of this is merely the sum of RISE mask generation
        # normalized in the [-1,1] range as the generation stage does nothing
        # given the constant blackbox response.
        inst = MCRISEStack(n=5, s=8, p1=0.5, fill_colors=[[255, 255, 255], [255, 0, 0], [0, 0, 255]], seed=0)
        # Results may be sensitive to changes in scikit-image. Version 0.19
        # introduces some changes to the resize function. Difference is
        # expected to only be marginal.
        res = inst.generate(test_image, test_bb)

        snapshot_custom.assert_match(res)
