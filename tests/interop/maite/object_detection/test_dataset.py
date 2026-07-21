from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from maite.protocols import DatumMetadata
from PIL import Image  # type: ignore
from typing_extensions import ReadOnly

from tests import DATA_DIR
from xaitk_saliency.interop.maite.object_detection.dataset import (
    MAITEDetectionTarget,
    MAITEObjectDetectionDataset,
)

kwcoco = None
COCOMAITEObjectDetectionDataset = None
try:
    import kwcoco  # type: ignore

    from xaitk_saliency.interop.maite.object_detection.dataset import (
        COCOMAITEObjectDetectionDataset,
    )

    is_usable = True
except ImportError:
    is_usable = False

rng = np.random.default_rng()
dset_dir = Path(DATA_DIR)


class _DummyDatumMetadata(DatumMetadata):
    foo: ReadOnly[str]


@pytest.mark.skipif(not is_usable, reason="Extra 'xaitk-saliency[tools]' not installed.")
class TestCOCOMAITEObjectDetectionDataset:
    if is_usable and kwcoco is not None:
        coco_file = dset_dir / "annotations.json"
        kwcoco_dataset = kwcoco.CocoDataset(coco_file)  # type: ignore
        metadata = [
            _DummyDatumMetadata(id=1, foo="a"),
            _DummyDatumMetadata(id=2, foo="b"),
            _DummyDatumMetadata(id=3, foo="c"),
        ]

        test_img_files = ["test_image1.png", "test_image2.png"]
        test_imgs = [np.array(Image.open(dset_dir / f)) for f in test_img_files]
        # Convert grayscale format from (H,W) to (H,W,C)
        test_imgs[1] = np.expand_dims(test_imgs[1], axis=2)

    test_bboxes = [
        np.array(
            [
                [10.0, 20.0, 40.0, 60.0],
                [12.0, 45.0, 62.0, 75.0],
                [30.0, 5.0, 77.0, 97.0],
            ],
        ),
        np.array([[0.0, 0.0, 5.0, 5.0], [50.0, 50.0, 82.0, 67.0], [68.0, 82.0, 79.0, 89.0]]),
    ]

    test_scores = [np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 0.5])]

    def test_len(self) -> None:
        """Test length property."""
        if TYPE_CHECKING:
            assert COCOMAITEObjectDetectionDataset is not None
        dataset = COCOMAITEObjectDetectionDataset(
            kwcoco_dataset=TestCOCOMAITEObjectDetectionDataset.kwcoco_dataset,
            image_metadata=TestCOCOMAITEObjectDetectionDataset.metadata,
        )
        assert len(dataset) == len(TestCOCOMAITEObjectDetectionDataset.test_imgs)

    def test_get_item(self) -> None:
        """Test indexing."""
        if TYPE_CHECKING:
            assert COCOMAITEObjectDetectionDataset is not None

        dataset = COCOMAITEObjectDetectionDataset(
            kwcoco_dataset=TestCOCOMAITEObjectDetectionDataset.kwcoco_dataset,
            image_metadata=TestCOCOMAITEObjectDetectionDataset.metadata,
        )

        for idx in range(len(dataset)):
            img, dets, md = dataset[idx]
            # assert False
            assert np.array_equal(
                img,
                np.transpose(np.asarray(TestCOCOMAITEObjectDetectionDataset.test_imgs[idx]), axes=(2, 0, 1)),
            )
            assert np.array_equal(dets.boxes, TestCOCOMAITEObjectDetectionDataset.test_bboxes[idx])
            assert np.array_equal(dets.scores, TestCOCOMAITEObjectDetectionDataset.test_scores[idx])
            # -1 because these IDs start at 1
            assert "foo" in md
            assert md["foo"] == TestCOCOMAITEObjectDetectionDataset.metadata[md["id"] - 1]["foo"]

    def test_bad_metadata(self) -> None:
        """Test that an exception is appropriately raised if metadata is missing."""
        if TYPE_CHECKING:
            assert COCOMAITEObjectDetectionDataset is not None

        with pytest.raises(ValueError, match=r"Image metadata length mismatch"):
            _ = COCOMAITEObjectDetectionDataset(
                kwcoco_dataset=TestCOCOMAITEObjectDetectionDataset.kwcoco_dataset,
                image_metadata=list(),
            )


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
