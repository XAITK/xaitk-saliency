from __future__ import annotations

from collections.abc import Hashable, Sequence
from contextlib import AbstractContextManager
from unittest.mock import MagicMock

import maite.protocols.object_detection as od
import numpy as np
import pytest
from smqtk_core.configuration import configuration_test_helper
from syrupy.assertion import SnapshotAssertion

from xaitk_saliency.interop.maite.object_detection import MAITEDetector

rng = np.random.default_rng()


@pytest.mark.maite
class TestMAITEObjectDetector:
    dummy_id_to_name = {0: "A", 1: "B", 2: "C"}
    dummy_boxes = np.asarray([[1, 2, 3, 4], [1, 2, 3, 4], [5, 6, 7, 8]])
    dummy_scores = np.asarray([0.25, 0.75, 0.95])
    dummy_labels = np.asarray([0, 2, 0])
    dummy_out = MagicMock(
        spec=od.ObjectDetectionTarget,
        boxes=dummy_boxes,
        labels=dummy_labels,
        scores=dummy_scores,
    )

    dummy_multiclass_boxes = np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]])
    dummy_multiclass_scores = np.asarray([[0.25, 0, 0.75], [0.95, 0, 0]])
    dummy_multiclass_labels = np.asarray([0, 2, 0])
    dummy_multiclass_out = MagicMock(
        spec=od.ObjectDetectionTarget,
        boxes=dummy_multiclass_boxes,
        labels=dummy_multiclass_labels,
        scores=dummy_multiclass_scores,
    )

    @pytest.mark.parametrize(
        ("detector", "id_to_name", "img_batch_size", "expectation"),
        [
            (
                MagicMock(spec=od.Model),
                dummy_id_to_name,
                2,
                pytest.raises(NotImplementedError, match=r"Constructor arg"),
            ),
        ],
    )
    def test_configuration(
        self,
        detector: od.Model,
        id_to_name: dict[int, Hashable],
        img_batch_size: int,
        expectation: AbstractContextManager,
    ) -> None:
        """Test configuration stability."""
        inst = MAITEDetector(detector=detector, ids=list(id_to_name.keys()), img_batch_size=img_batch_size)
        with expectation:
            configuration_test_helper(inst)

    @pytest.mark.parametrize(
        ("detector_output", "id_to_name", "img_batch_size", "imgs"),
        [
            (
                [dummy_out],
                dummy_id_to_name,
                1,
                [rng.integers(0, 255, (3, 256, 256), dtype=np.uint8)],
            ),
            (
                [dummy_out],
                dummy_id_to_name,
                1,
                [rng.integers(0, 255, (256, 256), dtype=np.uint8)],
            ),
            (
                [dummy_out] * 2,
                dummy_id_to_name,
                2,
                rng.integers(0, 255, (2, 256, 256), dtype=np.uint8),
            ),
            (
                [MagicMock(spec=od.ObjectDetectionTarget, boxes=[], labels=[], scores=[])],
                dummy_id_to_name,
                1,
                rng.integers(0, 255, (1, 256, 256), dtype=np.uint8),
            ),
            (
                [dummy_multiclass_out],
                dummy_id_to_name,
                1,
                [rng.integers(0, 255, (3, 256, 256), dtype=np.uint8)],
            ),
        ],
        ids=["single 3 channel", "single greyscale", "multiple images", "no dets", "multiclass_scores"],
    )
    def test_smoketest(
        self,
        snapshot: SnapshotAssertion,
        detector_output: Sequence[od.ObjectDetectionTarget],
        id_to_name: dict[int, Hashable],
        img_batch_size: int,
        imgs: np.ndarray | Sequence[np.ndarray],
    ) -> None:
        """Test that MAITE detector output is transformed appropriately."""
        mock_detector = MagicMock(spec=od.Model, return_value=detector_output)

        inst = MAITEDetector(detector=mock_detector, ids=list(id_to_name.keys()), img_batch_size=img_batch_size)
        res = list(inst.detect_objects(imgs))
        res = [[(bbox, {k: float(v) for k, v in scores.items()}) for bbox, scores in image_dets] for image_dets in res]
        assert res == snapshot
