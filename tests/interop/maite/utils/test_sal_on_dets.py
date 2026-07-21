import unittest.mock as mock
from collections.abc import Hashable, Iterable, Sequence
from typing import TypedDict
from unittest.mock import MagicMock

import numpy as np
from maite.protocols.object_detection import Dataset, Model
from smqtk_core.configuration import to_config_dict
from smqtk_detection.interfaces.detect_image_objects import DetectImageObjects
from smqtk_image_io.bbox import AxisAlignedBoundingBox
from typing_extensions import ReadOnly, Required

from xaitk_saliency.impls.gen_object_detector_blackbox_sal.drise import DRISEStack
from xaitk_saliency.interop.maite.object_detection.dataset import (
    MAITEDetectionTarget,
    MAITEObjectDetectionDataset,
)
from xaitk_saliency.interop.maite.object_detection.model import MAITEDetector
from xaitk_saliency.interop.maite.utils.sal_on_dets import compute_sal_maps, sal_on_dets

rng = np.random.default_rng()


class _DummyDatumMetadata(TypedDict):
    id: Required[ReadOnly[int]]


class TestComputeSalMaps:
    def test_compute_sal_maps(self) -> None:
        """Test saliency map generation with dummy detector, RISEGrid, and DRISEScoring."""

        class TestDetector(DetectImageObjects):
            """Dummy detector that returns consant detections."""

            def detect_objects(
                self,
                img_iter: Iterable[np.ndarray],
            ) -> Iterable[Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]]:
                for _ in img_iter:
                    yield [
                        (
                            AxisAlignedBoundingBox((0, 0), (10, 15)),
                            {"c0": 0.1, "c1": 0.7, "c2": 0.2},
                        ),
                        (
                            AxisAlignedBoundingBox((5, 5), (20, 25)),
                            {"c0": 0.85, "c1": 0.1, "c2": 0.05},
                        ),
                    ]

            get_config = None  # type: ignore

        sal_generator = DRISEStack(n=1, s=3, p1=0.5)
        detector = TestDetector()
        dataset = MAITEObjectDetectionDataset(
            imgs=[rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)] * 4,
            dets=[
                MAITEDetectionTarget(
                    boxes=np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]]),
                    labels=np.asarray([0, 1]),
                    scores=np.asarray([0.85, 0.64]),
                ),
            ]
            * 4,
            datum_metadata=[_DummyDatumMetadata(id=idx) for idx in range(4)],
            dataset_id="dummy_dataset",
        )

        sal_maps, sal_md = compute_sal_maps(
            dataset=dataset,
            sal_generator=sal_generator,
            blackbox_detector=detector,
            num_classes=3,
        )

        assert len(sal_maps) == len(dataset)
        for sal_map in sal_maps:
            assert len(sal_map) == 2  # 2 detections per image in this test
        assert sal_md == to_config_dict(sal_generator)


class TestSalOnDets:
    @mock.patch("xaitk_saliency.interop.maite.utils.sal_on_dets.compute_sal_maps", return_value=(list(), dict()))
    def test_sal_on_dets(self, patch: MagicMock) -> None:
        """Test workflow with MAITE detector."""
        dataset = MagicMock(spec=Dataset)
        sal_generator = DRISEStack(n=1, s=3, p1=0.5)
        maite_detector = MagicMock(spec=Model)
        ids: Sequence[int] = [0, 1, 2]
        img_batch_size = 4

        sal_on_dets(
            dataset=dataset,
            sal_generator=sal_generator,
            detector=maite_detector,
            ids=ids,
            img_batch_size=img_batch_size,
        )

        # Confirm compute_sal_maps arguments are as expected
        kwargs = patch.call_args.kwargs
        assert kwargs["dataset"] == dataset
        assert kwargs["sal_generator"] == sal_generator
        assert isinstance(kwargs["blackbox_detector"], MAITEDetector)
        assert kwargs["blackbox_detector"]._detector == maite_detector
        assert kwargs["blackbox_detector"]._ids == ids
        assert kwargs["blackbox_detector"]._img_batch_size == img_batch_size
        assert kwargs["num_classes"] == len(ids)
