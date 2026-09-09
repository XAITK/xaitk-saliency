"""Defines ``DatamaiteUtils``. Requires the ``tools`` extra (``datamaite``)."""

from __future__ import annotations

import logging
from collections.abc import Generator
from typing import TYPE_CHECKING, Any

import numpy as np
from datamaite.geometry import to_xyxy

if TYPE_CHECKING:
    from datamaite import ImageObjectDetectionSample, ObjectDetectionAnnotation, ObjectDetectionDataset

LOG = logging.getLogger(__name__)


class DatamaiteUtils:
    """Class for datamaite utility functions."""

    @classmethod
    def _detection_score(
        cls,
        det: ObjectDetectionAnnotation,
        *,
        num_classes: int,
        source_to_dense_id: dict[int | str | None, int],
    ) -> np.ndarray[Any, Any] | None:
        """Compute a detection's score vector, or `None` if it has no usable score.

        :param det: Detection to compute a score for.
        :param num_classes: Number of categories declared in the dataset's taxonomy.
        :param source_to_dense_id: Mapping from source category id to dense index,
            produced by `taxonomy.dense_ids()`.

        :return: A `[nClasses]` score vector, or `None` if `det` has no usable score.
        """
        prob = det.attributes.get("prob")
        if prob is not None:
            return np.asarray(prob, dtype=float)

        score = det.score if det.score is not None else det.attributes.get("score")
        if score is None:
            return None

        dense_id = source_to_dense_id.get(det.category_id)
        if dense_id is None:
            return None

        score_array = np.zeros(num_classes)
        score_array[dense_id] = score
        return score_array

    @classmethod
    def _parse_sample(
        cls,
        sample: ImageObjectDetectionSample,
        *,
        num_classes: int,
        source_to_dense_id: dict[int | str | None, int],
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], list[ObjectDetectionAnnotation]] | None:
        """Build a sample's bounding box and score matrices, or `None` if it has no detections.

        Detections with no usable score are dropped.

        :param sample: Sample to build bounding box and score matrices for.
        :param num_classes: Number of categories declared in the dataset's taxonomy.
        :param source_to_dense_id: Mapping from source category id to dense index,
            produced by `taxonomy.dense_ids()`.

        :return: `(bboxes, scores, kept)` for `sample`, or `None` if it has no detections.
        """
        if not sample.detections:
            return None

        bboxes = np.empty((0, 4))
        scores = np.empty((0, num_classes))
        kept: list[ObjectDetectionAnnotation] = []

        for det in sample.detections:
            score_array = cls._detection_score(det, num_classes=num_classes, source_to_dense_id=source_to_dense_id)
            if score_array is None:
                # no usable score
                continue

            scores = np.vstack((scores, score_array))
            bboxes = np.vstack((bboxes, to_xyxy(det.bbox)))
            kept.append(det)

        return bboxes, scores, kept

    def parse_datamaite_dset(
        self,
        dataset: ObjectDetectionDataset,
    ) -> Generator[
        tuple[
            np.ndarray[Any, Any],
            np.ndarray[Any, Any],
            np.ndarray[Any, Any],
            ImageObjectDetectionSample,
            list[ObjectDetectionAnnotation],
        ],
        None,
        None,
    ]:
        """Generate reference image, bounding box, and class score matrices.

        This is for use with an implementation of `GenerateObjectDetectorBlackboxSaliency`,
        from a `datamaite.ObjectDetectionDataset` object.
        Matrices are generated for each image in the dataset that has
        annotations associated with it.

        The output bounding boxes matrices are each of shape [nDets x 4].
        The output score matrices are each of shape [nDets x nClasses].
        Source category ids are not guaranteed contiguous or 0-based, so
        each detection's category id is remapped to its dense position in the
        dataset's taxonomy; nClasses is the number of categories declared in
        that taxonomy.

        We also yield each source image's `ImageObjectDetectionSample` and
        the list of `ObjectDetectionAnnotation` objects kept for it. This
        prevents us from having to re-decode images or re-derive what detections
        were kept.

        :param dataset: `datamaite.ObjectDetectionDataset` object to parse.

        :raises ValueError: If the dataset has no declared category taxonomy.

        :return: A generator of reference images, corresponding bounding boxes, scores as
            would be input to an implementation of `GenerateObjectDetectorBlackboxSaliency`,
            the source image sample, and the kept detections.
        """
        taxonomy = dataset.dataset_metadata.taxonomy
        if taxonomy is None or not taxonomy.entries:
            raise ValueError(f"No categories declared in dataset {dataset.dataset_id}.")
        source_to_dense_id = taxonomy.dense_ids()
        num_classes = len(taxonomy.entries)

        num_dropped = 0
        for idx, sample in enumerate(dataset.samples):
            parsed = self._parse_sample(sample, num_classes=num_classes, source_to_dense_id=source_to_dense_id)
            if parsed is None:
                # filter out images with no detections
                continue
            bboxes, scores, kept = parsed
            num_dropped += len(sample.detections) - len(kept)

            ref_img = np.transpose(dataset.get_input(idx), (1, 2, 0))

            yield ref_img, bboxes, scores, sample, kept

        if num_dropped:
            LOG.warning(
                f"Dropped {num_dropped} detections with no usable score or an undeclared category id.",
            )
