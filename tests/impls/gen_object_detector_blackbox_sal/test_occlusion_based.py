import numpy as np
from typing import Iterable, Tuple, Dict, Any, Hashable
import unittest.mock as mock
import gc

from smqtk_detection import DetectImageObjects
from smqtk_detection.utils.bbox import AxisAlignedBoundingBox
from smqtk_core.configuration import configuration_test_helper

from xaitk_saliency import PerturbImage, GenerateDetectorProposalSaliency
from xaitk_saliency.impls.gen_object_detector_blackbox_sal.occlusion_based import PerturbationOcclusion
from xaitk_saliency.utils.masking import occlude_image_batch


class TestPerturbationOcclusion:

    def teardown(self) -> None:
        # Collect any temporary implementations so they are not returned during
        # later `*.get_impl()` requests.
        gc.collect()

    def test_configuration(self) -> None:
        """
        Test configuration suite using known simple implementations.
        """

        class StubPI (PerturbImage):
            perturb = None  # type: ignore

            def __init__(self, stub_param: int):
                self.p = stub_param

            def get_config(self) -> Dict[str, Any]:
                return {'stub_param': self.p}

        class StubGen (GenerateDetectorProposalSaliency):
            generate = None  # type: ignore

            def __init__(self, stub_param: int):
                self.p = stub_param

            def get_config(self) -> Dict[str, Any]:
                return {'stub_param': self.p}

        test_threads = 87
        test_spi_p = 0
        test_sgn_p = 1
        inst = PerturbationOcclusion(
            StubPI(test_spi_p), StubGen(test_sgn_p), 87
        )
        for inst_i in configuration_test_helper(inst):
            assert inst_i._threads == test_threads
            assert isinstance(inst_i._perturber, StubPI)
            assert inst_i._perturber.p == test_spi_p
            assert isinstance(inst_i._generator, StubGen)
            assert inst_i._generator.p == test_sgn_p

    def test_generate_success(self) -> None:
        """
        Test successfully invoking _generate().
        """

        class StubPI (PerturbImage):
            """
            Stub perturber that returns masks of ones.
            """

            def perturb(self, ref_image: np.ndarray) -> np.ndarray:
                return np.ones((6, *ref_image.shape[:2]), dtype=bool)

            get_config = None  # type: ignore

        class StubGen (GenerateDetectorProposalSaliency):
            """
            Stub saliency generator that returns zeros with correct shape.
            """

            def generate(
                self,
                ref_dets: np.ndarray,
                pert_dets: np.ndarray,
                pert_masks: np.ndarray
            ) -> np.ndarray:
                return np.zeros((ref_dets.shape[0], *pert_masks.shape[1:]), dtype=np.float16)

            get_config = None  # type: ignore

        class StubDetector (DetectImageObjects):
            """
            Stub object detector that returns known detections.
            """

            def detect_objects(
                self,
                img_iter: Iterable[np.ndarray]
            ) -> Iterable[Iterable[Tuple[AxisAlignedBoundingBox, Dict[Hashable, float]]]]:
                for i, _ in enumerate(img_iter):
                    # Return different number of detections for each image to
                    # test padding functinality
                    yield [(
                        AxisAlignedBoundingBox((0, 0), (1, 1)),
                        {'class0': 0.0, 'class1': 0.9}
                    ) for _ in range(i)]

            get_config = None  # type: ignore

        test_pi = StubPI()
        test_gen = StubGen()
        test_detector = StubDetector()

        test_image = np.ones((64, 64, 3), dtype=np.uint8)

        test_bboxes = np.ones((3, 4))
        test_scores = np.ones((3, 2))

        # Call with default fill
        with mock.patch(
            'xaitk_saliency.impls.gen_object_detector_blackbox_sal.occlusion_based.occlude_image_batch',
            wraps=occlude_image_batch
        ) as m_occ_img:
            inst = PerturbationOcclusion(test_pi, test_gen)
            test_result = inst._generate(
                test_image,
                test_bboxes,
                test_scores,
                test_detector
            )

            assert test_result.shape == (3, 64, 64)
            # The "fill" kwarg passed to occlude_image_batch should match
            # the default given, which is None
            m_occ_img.assert_called_once()
            # Using [-1] indexing for compatibility with python 3.7
            m_kwargs = m_occ_img.call_args[-1]
            assert "fill" in m_kwargs
            assert m_kwargs['fill'] is None

        # Call with a specified fill value
        test_fill = [123, 21, 42]
        with mock.patch(
            'xaitk_saliency.impls.gen_object_detector_blackbox_sal.occlusion_based.occlude_image_batch',
            wraps=occlude_image_batch
        ) as m_occ_img:
            inst = PerturbationOcclusion(test_pi, test_gen)
            inst.fill = test_fill
            test_result = inst._generate(
                test_image,
                test_bboxes,
                test_scores,
                test_detector
            )

            assert test_result.shape == (3, 64, 64)
            # The "fill" kwarg passed to occlude_image_batch should match
            # the attribute above
            m_occ_img.assert_called_once()
            # Using [-1] indexing for compatibility with python 3.7
            m_kwargs = m_occ_img.call_args[-1]
            assert "fill" in m_kwargs
            assert m_kwargs['fill'] == test_fill
