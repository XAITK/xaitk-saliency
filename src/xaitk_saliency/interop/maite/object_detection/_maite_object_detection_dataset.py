"""This module provides dataset classes for working with object detection data in the MAITE framework.

Classes:
    MAITEDetectionTarget: A dataclass for storing detection results, including bounding boxes, labels, and scores.
    MAITEObjectDetectionDataset: A generic wrapper for object detection datasets with varying-sized images.

Dependencies:
    - numpy: For numerical operations.
    - maite.protocols: For object detection protocols.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from maite.protocols import DatasetMetadata
from maite.protocols.object_detection import (
    Dataset,
    DatumMetadataType,
    InputType,
    TargetType,
)
from typing_extensions import override

OBJ_DETECTION_DATUM_T = tuple[InputType, TargetType, DatumMetadataType]


@dataclass
class MAITEDetectionTarget:
    """Dataclass for the datum-level MAITE output detection format."""

    boxes: np.ndarray[Any, Any]
    labels: np.ndarray[Any, Any]
    scores: np.ndarray[Any, Any]


class MAITEObjectDetectionDataset(Dataset):
    """Implementation of the MAITE Object Detection dataset wrapper for dataset images of varying sizes.

    Parameters
    ----------
    imgs : Sequence[np.ndarray]
        Sequence of images.
    dets : Sequence[ObjectDetectionTarget]
        Sequence of detections for each image.
    datum_metadata : Sequence[DatumMetadataType]
        Sequence of metadata for each image.
    dataset_id: str
        Dataset ID.
    index2label: dict[int, str] | None
        Mapping from class index to label.
    """

    def __init__(
        self,
        *,
        imgs: Sequence[np.ndarray[Any, Any]],
        dets: Sequence[TargetType],
        datum_metadata: Sequence[DatumMetadataType],
        dataset_id: str,
        index2label: dict[int, str] | None = None,
    ) -> None:
        """Initialize MAITE-compliant dataset.

        Args:
            imgs (Sequence[np.ndarray]): Sequence of images in the dataset.
            dets (Sequence[TargetType]): Sequence of detection targets for the images.
            datum_metadata (Sequence[DatumMetadataType]): Sequence of metadata dictionaries.
            dataset_id (str): Dataset ID.
            index2label (dict[int, str] | None): Mapping from class index to label.
        """
        self.imgs = imgs
        self.dets = dets
        self.datum_metadata = datum_metadata
        if index2label is not None:
            self.metadata: DatasetMetadata = {
                "id": dataset_id,
                "index2label": index2label,
            }
        else:
            self.metadata = {
                "id": dataset_id,
            }

    @override
    def __len__(self) -> int:
        """Returns the number of images in the dataset."""
        return len(self.imgs)

    @override
    def __getitem__(self, index: int) -> OBJ_DETECTION_DATUM_T:
        """Returns the dataset object at the given index."""
        return self.imgs[index], self.dets[index], self.datum_metadata[index]
