from __future__ import annotations

from collections.abc import Hashable, Sequence
from contextlib import AbstractContextManager
from unittest.mock import MagicMock

import maite.protocols.image_classification as ic
import numpy as np
import pytest
from smqtk_core.configuration import configuration_test_helper
from syrupy.assertion import SnapshotAssertion

from xaitk_saliency.interop.maite.image_classification import MAITEImageClassifier

rng = np.random.default_rng()


@pytest.mark.maite
class TestMAITEImageClassifier:
    dummy_id_to_name_1 = {0: "A", 1: "B", 2: "C"}
    expected_labels = [0, 1, 2]
    dummy_id_to_name_2 = {0: "D", 2: "D", 1: "E"}

    dummy_non_consecutive_id_to_name_1 = {1: "B", 2: "C", 4: "E"}
    dummy_non_consecutive_id_to_name_2 = {1: "B", 4: "E", 2: "B"}
    expected_non_consecutive_labels = [1, 2, 4]

    dummy_out = np.asarray([0, 0.15, 0.8])

    @pytest.mark.parametrize(
        ("classifier", "id_to_name", "img_batch_size", "expectation"),
        [
            (
                MagicMock(spec=ic.Model),
                dummy_id_to_name_1,
                3,
                pytest.raises(NotImplementedError, match=r"Constructor arg"),
            ),
        ],
    )
    def test_configuration(
        self,
        classifier: ic.Model,
        id_to_name: dict[int, Hashable],
        img_batch_size: int,
        expectation: AbstractContextManager,
    ) -> None:
        """Test configuration stability."""
        inst = MAITEImageClassifier(classifier=classifier, ids=list(id_to_name.keys()), img_batch_size=img_batch_size)
        with expectation:
            configuration_test_helper(inst)

    @pytest.mark.parametrize(
        (
            "classifier_output",
            "id_to_name",
            "img_batch_size",
            "imgs",
        ),
        [
            (
                [dummy_out],
                dummy_id_to_name_1,
                1,
                [rng.integers(0, 255, (3, 256, 256), dtype=np.uint8)],
            ),
            (
                [dummy_out],
                dummy_id_to_name_1,
                1,
                [rng.integers(0, 255, (256, 256), dtype=np.uint8)],
            ),
            (
                [dummy_out] * 2,
                dummy_id_to_name_1,
                2,
                rng.integers(0, 255, (2, 256, 256), dtype=np.uint8),
            ),
            (
                [dummy_out],
                dummy_non_consecutive_id_to_name_1,
                1,
                [rng.integers(0, 255, (3, 256, 256), dtype=np.uint8)],
            ),
            (
                [dummy_out],
                dummy_non_consecutive_id_to_name_2,
                1,
                [rng.integers(0, 255, (3, 256, 256), dtype=np.uint8)],
            ),
        ],
        ids=[
            "single 3 channel",
            "single greyscale",
            "multiple images",
            "non-consecutive labels",
            "unsorted non-consecutive labels",
        ],
    )
    def test_smoketest(
        self,
        snapshot: SnapshotAssertion,
        classifier_output: ic.TargetType,
        id_to_name: dict[int, Hashable],
        img_batch_size: int,
        imgs: np.ndarray | Sequence[np.ndarray],
    ) -> None:
        """Test that MAITE classifier output is transformed appropriately."""
        mock_classifier = MagicMock(spec=ic.Model, return_value=classifier_output)

        inst = MAITEImageClassifier(
            classifier=mock_classifier,
            ids=list(id_to_name.keys()),
            img_batch_size=img_batch_size,
        )

        res = list(inst.classify_images(imgs))
        res = [{k: float(v) for k, v in d.items()} for d in res]
        assert res == snapshot

    @pytest.mark.parametrize(
        ("id_to_name"),
        [
            (dummy_id_to_name_1),
            (dummy_id_to_name_2),
            (dummy_non_consecutive_id_to_name_1),
            (dummy_non_consecutive_id_to_name_2),
        ],
    )
    def test_labels(
        self,
        snapshot: SnapshotAssertion,
        id_to_name: dict[int, Hashable],
    ) -> None:
        """Test that get_labels() returns the correct labels."""
        inst = MAITEImageClassifier(classifier=MagicMock(spec=ic.Model), ids=list(id_to_name.keys()))

        assert inst.get_labels() == snapshot
