import numpy as np
import pytest
from maite.protocols import DatumMetadata
from typing_extensions import ReadOnly

from xaitk_saliency.interop.maite.object_detection import (
    MAITEDetectionTarget,
    MAITEObjectDetectionDataset,
)

rng = np.random.default_rng()


class _DummyDatumMetadata(DatumMetadata):
    foo: ReadOnly[str]


@pytest.mark.maite
class TestMAITEObjectDetectionDataset:
    imgs = [rng.integers(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(2)]
    dets = [
        MAITEDetectionTarget(
            boxes=np.asarray([[0.0, 0.0, 50.0, 50.0]]),
            labels=np.asarray([0]),
            scores=np.asarray([0.25]),
        ),
        MAITEDetectionTarget(
            boxes=np.asarray([[0.0, 0.0, 3.0, 4.0], [5.6, 10.7, 52.3, 76.0]]),
            labels=np.asarray([1, 0]),
            scores=np.asarray([0.34, 0.95]),
        ),
    ]
    metadata = [
        _DummyDatumMetadata(id=1, foo="a"),
        _DummyDatumMetadata(id=2, foo="b"),
    ]

    def test_len(self) -> None:
        """Test length property."""
        dataset = MAITEObjectDetectionDataset(
            imgs=TestMAITEObjectDetectionDataset.imgs,
            dets=TestMAITEObjectDetectionDataset.dets,
            datum_metadata=TestMAITEObjectDetectionDataset.metadata,
            dataset_id="dummy_dataset",
        )

        assert len(dataset) == len(TestMAITEObjectDetectionDataset.imgs)

    def test_get_item(self) -> None:
        """Test indexing."""
        dataset = MAITEObjectDetectionDataset(
            imgs=TestMAITEObjectDetectionDataset.imgs,
            dets=TestMAITEObjectDetectionDataset.dets,
            datum_metadata=TestMAITEObjectDetectionDataset.metadata,
            dataset_id="dummy_dataset",
        )

        for idx in range(len(dataset)):
            img, dets, md = dataset[idx]
            assert np.array_equal(img, TestMAITEObjectDetectionDataset.imgs[idx])
            assert dets == TestMAITEObjectDetectionDataset.dets[idx]
            assert md == TestMAITEObjectDetectionDataset.metadata[idx]
