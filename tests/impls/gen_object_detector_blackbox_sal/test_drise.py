from collections.abc import Hashable, Iterable

import numpy as np
from smqtk_core.configuration import configuration_test_helper
from smqtk_detection.interfaces.detect_image_objects import DetectImageObjects
from smqtk_image_io.bbox import AxisAlignedBoundingBox

from tests import DATA_DIR
from xaitk_saliency.impls.gen_object_detector_blackbox_sal.drise import DRISEScoring, DRISEStack, RISEGrid


class TestBlackBoxDRISE:
    def test_configuration(self) -> None:
        """Test configuration suite."""

        inst = DRISEStack(
            n=123,
            s=8,
            p1=0.73,
            seed=98,
            threads=5,
        )

        for inst_i in configuration_test_helper(inst):
            inst_p = inst_i._po._perturber
            inst_g = inst_i._po._generator

            assert isinstance(inst_p, RISEGrid)
            assert isinstance(inst_g, DRISEScoring)
            assert inst_p.n == 123
            assert inst_p.s == 8
            assert np.allclose(inst_p.p1, 0.73)
            assert inst_p.seed == 98
            assert inst_p.threads == 5
            assert inst_g.get_config() == {}

    def test_generation_rgb(self) -> None:
        """Test basic generation functionality with dummy inputs"""

        class TestDetector(DetectImageObjects):
            """Dummy detector that returns consant detections."""

            def detect_objects(
                self,
                img_iter: Iterable[np.ndarray],
            ) -> Iterable[Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]]:
                for _ in img_iter:
                    yield [
                        (AxisAlignedBoundingBox((0, 0), (10, 15)), {"class0": 0.3, "class1": 0.7}),
                        (AxisAlignedBoundingBox((5, 5), (20, 25)), {"class0": 0.9, "class1": 0.1}),
                    ]

            get_config = None  # type: ignore

        test_det = TestDetector()
        test_image = np.full([32, 32, 3], fill_value=255, dtype=np.uint8)

        test_bboxes = np.array([[4, 2, 22, 30], [15, 8, 31, 16]])

        test_scores = np.array([[0.1, 0.9], [0.5, 0.5]])

        inst = DRISEStack(n=5, s=8, p1=0.5, seed=0)
        res = inst.generate(test_image, test_bboxes, test_scores, test_det)

        exp_res = np.load(DATA_DIR / "exp_drise_stack_res.npy")
        assert np.allclose(exp_res, res)

    def test_fill_prop(self) -> None:
        """
        Test that the `fill` property appropriately gets and sets the
        underlying `PerturbationOcclusion` instance fill instance attribute.
        """
        inst = DRISEStack(5, 8, 0.5, seed=0)
        assert inst._po.fill is None
        assert inst.fill is None
        inst.fill = 42
        assert inst.fill == 42
        assert inst._po.fill == 42
