"""This module provides the `MAITEDetector` class, an adapter for the MAITE object detection protocol.

It converts MAITE model outputs into the SMQTK `DetectImageObjects` interface format, making
it compatible with downstream detection pipelines.

Classes:
    MAITEDetector: Adapts a MAITE object detection model to the SMQTK interface.

Dependencies:
    - maite.protocols.object_detection: For the MAITE object detection model protocol.
    - numpy: For numerical computations.
    - smqtk_detection: For the SMQTK object detection interface.
    - smqtk_image_io.AxisAlignedBoundingBox: For handling bounding boxes.
"""

from collections.abc import Hashable, Iterable, Sequence

import maite.protocols.object_detection as od
import numpy as np
from smqtk_detection.interfaces.detect_image_objects import DetectImageObjects
from smqtk_image_io.bbox import AxisAlignedBoundingBox
from typing_extensions import override


class MAITEDetector(DetectImageObjects):
    """Adapter for MAITE object detection protocol.

    It transforms its outputs into the SMQTK `DetectImageObjects` interface format.

    Attributes:
        _detector (od.Model): The MAITE protocol-based detector instance.
        _ids (Dict[int, Hashable]): Mapping from label IDs to human-readable names.
        _img_batch_size (int): The number of images to process in a single batch.

    Methods:
        detect_objects: Detect objects in a batch of images and return results in SMQTK format.
        get_config: Raises `NotImplementedError` (configuration serialization is not implemented).
    """

    def __init__(
        self,
        detector: od.Model,
        ids: Sequence[int],
        img_batch_size: int = 1,
    ) -> None:
        """Initialize the MAITEDetector with a MAITE protocol-based object detector.

        Args:
            detector (od.Model): The MAITE object detection model.
            ids (Dict[int, Hashable]): A dictionary mapping label IDs to human-readable names.
            img_batch_size (int, optional): The number of images to process in a single batch. Defaults to 1.
        """
        self._detector = detector
        self._ids = sorted(ids)
        self._img_batch_size = img_batch_size

    @override
    def detect_objects(  # noqa: C901
        self,
        img_iter: Iterable[np.ndarray],
    ) -> Iterable[Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]]:
        all_out = list()

        def _to_channels_first(img: np.ndarray) -> np.ndarray:
            """Convert an image from channels-last format (H, W, C) to channels-first format (C, H, W).

            Args:
                img (np.ndarray): Input image.

            Returns:
                np.ndarray: The image in channels-first format.
            """
            if img.ndim < 3:
                return img
            return np.moveaxis(img, -1, 0)

        def _xform_dets(
            bboxes: Iterable[AxisAlignedBoundingBox],
            labels: np.ndarray,
            probs: np.ndarray,
        ) -> Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]:
            """Transform detection outputs into the SMQTK detection format.

            Args:
                bboxes (Iterable[AxisAlignedBoundingBox]): Detected bounding boxes.
                labels (Sequence[Hashable]): Detected class labels.
                probs (np.ndarray): Confidence scores for each detection.

            Returns:
                Iterable[Tuple[AxisAlignedBoundingBox, Dict[Hashable, float]]]:
                    A list of transformed detections, where each detection consists of:
                    - An `AxisAlignedBoundingBox`.
                    - A dictionary of class label-to-confidence mappings.
            """
            dets_dict: dict[AxisAlignedBoundingBox, dict[Hashable, float]] = dict()
            for box, label, prob in zip(bboxes, labels, probs, strict=False):
                if probs.ndim > 1:  # Scores per classes
                    dets_dict[box] = dict(zip(self._ids, prob, strict=False))
                else:  # Single score per box
                    if box not in dets_dict:
                        dets_dict[box] = dict.fromkeys(self._ids, 0.0)

                    dets_dict[box][label] = prob

            return list(dets_dict.items())

        def _xform_bbox(box: np.ndarray) -> AxisAlignedBoundingBox:
            """Convert a bounding box array into an `AxisAlignedBoundingBox` instance.

            Args:
                box (np.ndarray): A bounding box in `[x_min, y_min, x_max, y_max]` format.

            Returns:
                AxisAlignedBoundingBox: The corresponding `AxisAlignedBoundingBox` instance.
            """
            return AxisAlignedBoundingBox(box[0:2], box[2:4])

        def _generate_outputs(batch: Sequence[np.ndarray]) -> None:
            """Generate detection outputs for a batch of images.

            Args:
                batch (Sequence[np.ndarray]): A batch of images in channels-first format.

            Side Effects:
                Appends detection results to the `all_out` attribute.
            """
            predictions = self._detector(batch)

            for pred in predictions:
                boxes = [_xform_bbox(box) for box in np.asarray(pred.boxes)]
                labels = np.asarray(pred.labels)
                scores = np.asarray(pred.scores)

                all_out.append(_xform_dets(bboxes=boxes, labels=labels, probs=scores))

        # Batch model passes
        batch = list()
        for img in img_iter:
            batch.append(_to_channels_first(img))

            if len(batch) == self._img_batch_size:
                _generate_outputs(batch)
                batch = list()

        # Leftover batch
        if len(batch) > 0:
            _generate_outputs(batch)

        return all_out

    @override
    def get_config(self) -> dict:
        raise NotImplementedError(
            "Constructor arguments are not serializable as is and require further implementation to do so.",
        )
