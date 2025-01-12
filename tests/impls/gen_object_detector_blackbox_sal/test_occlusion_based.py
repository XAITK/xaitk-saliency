import gc
import unittest.mock as mock
from collections.abc import Hashable, Iterable
from typing import Any

import numpy as np
from smqtk_core.configuration import configuration_test_helper
from smqtk_detection import DetectImageObjects
from smqtk_detection.utils.bbox import AxisAlignedBoundingBox

from xaitk_saliency import GenerateDetectorProposalSaliency, PerturbImage
from xaitk_saliency.impls.gen_object_detector_blackbox_sal.occlusion_based import PerturbationOcclusion
from xaitk_saliency.utils.masking import occlude_image_batch


def _perturb(ref_image: np.ndarray) -> np.ndarray:
    return np.ones((6, *ref_image.shape[:2]), dtype=bool)


class StubPI(PerturbImage):
    perturb = None  # type: ignore

    def __init__(self, stub_param: int) -> None:
        self.p = stub_param

    def get_config(self) -> dict[str, Any]:
        return {"stub_param": self.p}


class StubGen(GenerateDetectorProposalSaliency):
    generate = None  # type: ignore

    def __init__(self, stub_param: int) -> None:
        self.p = stub_param

    def get_config(self) -> dict[str, Any]:
        return {"stub_param": self.p}


class TestPerturbationOcclusion:
    def teardown(self) -> None:
        # Collect any temporary implementations so they are not returned during
        # later `*.get_impl()` requests.
        gc.collect()  # pragma: no cover

    def test_configuration(self) -> None:
        """Test configuration suite using known simple implementations."""

        test_threads = 87
        test_spi_p = 0
        test_sgn_p = 1
        inst = PerturbationOcclusion(StubPI(test_spi_p), StubGen(test_sgn_p), threads=87)
        for inst_i in configuration_test_helper(inst):
            assert inst_i._threads == test_threads
            assert isinstance(inst_i._perturber, StubPI)
            assert inst_i._perturber.p == test_spi_p
            assert isinstance(inst_i._generator, StubGen)
            assert inst_i._generator.p == test_sgn_p

    def test_generate_success(self) -> None:
        """Test successfully invoking _generate()."""

        def detect_objects(
            img_iter: Iterable[np.ndarray],
        ) -> Iterable[Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]]:
            for i, _ in enumerate(img_iter):
                # Return different number of detections for each image to
                # test padding functinality
                yield [
                    (
                        AxisAlignedBoundingBox((0, 0), (1, 1)),
                        {"class0": 0.0, "class1": 0.9},
                    )
                    for _ in range(i)
                ]

        test_image = np.ones((64, 64, 3), dtype=np.uint8)

        test_bboxes = np.ones((3, 4))
        test_scores = np.ones((3, 2))

        m_perturb = mock.Mock(spec=PerturbImage)
        m_perturb.return_value = _perturb(test_image)
        m_gen = mock.Mock(spec=GenerateDetectorProposalSaliency)
        m_gen.return_value = np.zeros((3, 64, 64))
        m_detector = mock.Mock(spec=DetectImageObjects)
        m_detector.detect_objects = detect_objects

        # Call with default fill
        with mock.patch(
            "xaitk_saliency.impls.gen_object_detector_blackbox_sal.occlusion_based.occlude_image_batch",
            wraps=occlude_image_batch,
        ) as m_occ_img:
            inst = PerturbationOcclusion(m_perturb, m_gen)
            test_result = inst._generate(
                test_image,
                test_bboxes,
                test_scores,
                m_detector,
            )

            assert test_result.shape == (3, 64, 64)
            # The "fill" kwarg passed to occlude_image_batch should match
            # the default given, which is None
            m_occ_img.assert_called_once()
            # Using [-1] indexing for compatibility with python 3.7
            m_kwargs = m_occ_img.call_args[-1]
            assert "fill" in m_kwargs
            assert m_kwargs["fill"] is None

        # Call with a specified fill value
        test_fill = [123, 21, 42]
        with mock.patch(
            "xaitk_saliency.impls.gen_object_detector_blackbox_sal.occlusion_based.occlude_image_batch",
            wraps=occlude_image_batch,
        ) as m_occ_img:
            inst = PerturbationOcclusion(m_perturb, m_gen)
            inst.fill = test_fill
            test_result = inst._generate(
                test_image,
                test_bboxes,
                test_scores,
                m_detector,
            )

            assert test_result.shape == (3, 64, 64)
            # The "fill" kwarg passed to occlude_image_batch should match
            # the attribute above
            m_occ_img.assert_called_once()
            # Using [-1] indexing for compatibility with python 3.7
            m_kwargs = m_occ_img.call_args[-1]
            assert "fill" in m_kwargs
            assert m_kwargs["fill"] == test_fill

    def test_empty_detections(self) -> None:
        """Test invoking _generate() with empty detections."""

        def detect_objects(
            img_iter: Iterable[np.ndarray],
        ) -> Iterable[Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]]:
            for _i, _ in enumerate(img_iter):
                # Return 0 detections for each image
                yield []

        m_detector = mock.Mock(spec=DetectImageObjects)
        m_detector.detect_objects = detect_objects

        test_image = np.ones((64, 64, 3), dtype=np.uint8)

        test_bboxes = np.ones((3, 4))
        test_scores = np.ones((3, 2))

        m_perturb = mock.Mock(spec=PerturbImage)
        m_perturb.return_value = _perturb(test_image)
        m_gen = mock.Mock(spec=GenerateDetectorProposalSaliency)
        m_gen.return_value = np.zeros((3, 64, 64))
        m_detector = mock.Mock(spec=DetectImageObjects)
        m_detector.detect_objects = detect_objects

        inst = PerturbationOcclusion(m_perturb, m_gen)
        test_result = inst._generate(
            test_image,
            test_bboxes,
            test_scores,
            m_detector,
        )

        assert len(test_result) == 0
