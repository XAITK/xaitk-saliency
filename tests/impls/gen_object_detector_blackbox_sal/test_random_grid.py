import numpy as np
from typing import Iterable, Tuple, Dict, Hashable

from smqtk_detection import DetectImageObjects
from smqtk_detection.utils.bbox import AxisAlignedBoundingBox
from smqtk_core.configuration import configuration_test_helper

from xaitk_saliency.impls.gen_object_detector_blackbox_sal.drise import (
    RandomGrid,
    DRISEScoring,
    RandomGridStack
)


from tests import DATA_DIR


class TestBlackBoxRandomGrid:

    def test_configuration(self) -> None:
        """
        Test configuration suite.
        """

        inst = RandomGridStack(
            n=55,
            s=(15, 8),
            p1=0.34,
            seed=7,
            threads=2,
        )

        for inst_i in configuration_test_helper(inst):
            inst_p = inst_i._po._perturber
            inst_g = inst_i._po._generator

            assert isinstance(inst_p, RandomGrid)
            assert isinstance(inst_g, DRISEScoring)
            assert inst_p.n == 55
            assert inst_p.s == (15, 8)
            assert np.allclose(inst_p.p1, 0.34)
            assert inst_p.seed == 7
            assert inst_p.threads == 2
            assert inst_g.get_config() == {}

    def test_generation_rgb(self) -> None:
        """
        Test basic generation functionality with dummy inputs.
        """
        class TestDetector (DetectImageObjects):
            """
            Dummy detector that returns consant detections.
            """

            def detect_objects(
                self,
                img_iter: Iterable[np.ndarray]
            ) -> Iterable[Iterable[Tuple[AxisAlignedBoundingBox, Dict[Hashable, float]]]]:
                for _ in img_iter:
                    yield [
                        (AxisAlignedBoundingBox((2, 2), (10, 15)), {'class0': 0.2, 'class1': 0.8}),
                        (AxisAlignedBoundingBox((7, 7), (20, 25)), {'class0': 0.7, 'class1': 0.3}),
                        (AxisAlignedBoundingBox((9, 9), (15, 20)), {'class0': 0.5, 'class1': 0.5}),
                    ]

            get_config = None  # type: ignore

        test_det = TestDetector()
        test_image = np.full([32, 32, 3], fill_value=255, dtype=np.uint8)

        test_bboxes = np.array([
            [6, 5, 12, 19],
            [17, 18, 31, 22]
        ])

        test_scores = np.array([
            [0.9, 0.1],
            [0.4, 0.6]
        ])

        inst = RandomGridStack(
            n=5,
            s=(2, 8),
            p1=0.6,
            seed=42
        )
        res = inst.generate(
            test_image,
            test_bboxes,
            test_scores,
            test_det
        )

        exp_res = np.load(DATA_DIR / "exp_random_grid_stack_res.npy")
        assert np.allclose(exp_res, res)

    def test_fill_prop(self) -> None:
        """
        Test that the `fill` property appropriately gets and sets the
        underlying `PerturbationOcclusion` instance fill instance attribute.
        """
        inst = RandomGridStack(3, (4, 5), 0.6, seed=7)
        assert inst._po.fill is None
        assert inst.fill is None
        inst.fill = 24
        assert inst.fill == 24
        assert inst._po.fill == 24
