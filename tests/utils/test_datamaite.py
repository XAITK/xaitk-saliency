import os

import datamaite
import numpy as np
import pytest
from PIL import Image

from tests import DATA_DIR
from xaitk_saliency.utils.datamaite import DatamaiteUtils


@pytest.mark.tools
class TestParseDatamaiteDset:
    def test_dset_parse(self) -> None:
        """Test that a dummy detection file is parsed correctly."""
        test_dset = datamaite.load_od(
            DATA_DIR,
            dataset_format=datamaite.DatasetFormat.COCO,
            annotation_file="test_dets.json",
        )

        test_img_files = ["test_image1.png", "test_image2.png"]
        # datamaite's decoder always normalizes to 3-channel RGB so the images here
        # need the same normalization
        test_imgs = [np.array(Image.open(os.path.join(DATA_DIR, f)).convert("RGB")) for f in test_img_files]

        test_bboxes = [
            np.array([[10, 20, 40, 60], [12, 45, 62, 75], [30, 5, 77, 97]]),
            np.array([[50, 50, 82, 67], [68, 82, 79, 89]]),
        ]

        test_scores = [
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            np.array([[0.0, 1.0, 0.0], [0.2, 0.5, 0.3]]),
        ]

        # source annotation ids kept for each image, annotation 4 (test_image2, no
        # score/prob) is dropped
        test_det_ids = [[1, 2, 3], [5, 6]]

        inst = DatamaiteUtils()
        for i, (img, bboxes, scores, sample, kept) in enumerate(inst.parse_datamaite_dset(test_dset)):
            assert np.array_equal(img, test_imgs[i])
            assert np.array_equal(bboxes, test_bboxes[i])
            assert np.array_equal(scores, test_scores[i])
            assert sample.file_name == test_img_files[i]
            assert [det.source_annotation_id for det in kept] == test_det_ids[i]

    def test_no_categories(self) -> None:
        """Test that an exception is raised when the dataset has no declared category taxonomy."""
        empty_dset = datamaite.ObjectDetectionDataset(samples=())

        with pytest.raises(ValueError, match=r"No categories declared"):
            next(DatamaiteUtils().parse_datamaite_dset(empty_dset))

    def test_sample_without_detections_is_skipped(self) -> None:
        """Test that a sample with no detections is not yielded."""
        image_path = os.path.join(DATA_DIR, "test_image1.png")
        taxonomy = datamaite.Taxonomy(entries=(datamaite.CategoryEntry(source_id=1, name="cat"),))
        empty_sample = datamaite.ImageObjectDetectionSample(image_id=0, path_or_uri=image_path, file_name="empty.png")
        dset = datamaite.ObjectDetectionDataset(
            samples=(empty_sample,),
            dataset_metadata=datamaite.DatasetMetadata(taxonomy=taxonomy),
        )

        parsed = list(DatamaiteUtils().parse_datamaite_dset(dset))

        assert parsed == []

    def test_detection_without_score_is_dropped(self) -> None:
        """Test that a detection with no score is excluded from the parsed output."""
        image_path = os.path.join(DATA_DIR, "test_image1.png")
        taxonomy = datamaite.Taxonomy(entries=(datamaite.CategoryEntry(source_id=1, name="cat"),))
        sample = datamaite.ImageObjectDetectionSample(
            image_id=0,
            path_or_uri=image_path,
            file_name="sample.png",
            detections=(
                datamaite.ObjectDetectionAnnotation(bbox=(10, 20, 30, 40), category_id=1),
                datamaite.ObjectDetectionAnnotation(bbox=(50, 20, 30, 40), category_id=1, score=0.9),
            ),
        )
        dset = datamaite.ObjectDetectionDataset(
            samples=(sample,),
            dataset_metadata=datamaite.DatasetMetadata(taxonomy=taxonomy),
        )

        _, _, scores, _, kept = next(DatamaiteUtils().parse_datamaite_dset(dset))

        assert len(kept) == 1
        assert kept[0].score == pytest.approx(0.9)
        assert scores.shape == (1, 1)
