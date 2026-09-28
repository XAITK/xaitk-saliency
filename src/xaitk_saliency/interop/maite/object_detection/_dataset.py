"""This module provides dataset classes for working with object detection data in the MAITE framework.

It includes adapters for COCO-format datasets and general datasets for varying-sized images.

Classes:
    MAITEDetectionTarget: A dataclass for storing detection results, including bounding boxes, labels, and scores.
    COCOMAITEObjectDetectionDataset: Converts a COCO dataset to be compatible with the MAITE object detection protocol.
    MAITEObjectDetectionDataset: A generic wrapper for object detection datasets with varying-sized images.

Dependencies:
    - kwcoco: For working with COCO-format datasets.
    - numpy: For numerical operations.
    - maite.protocols: For object detection protocols.
    - PIL: For image processing.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from maite.protocols import DatasetMetadata, DatumMetadata
from maite.protocols.object_detection import (
    Dataset,
    DatumMetadataType,
    InputType,
    TargetType,
)
from PIL import Image
from typing_extensions import ReadOnly

try:
    import kwcoco  # noqa: F401

    is_usable: bool = True
except ImportError:
    is_usable = False

if TYPE_CHECKING:
    from kwcoco.coco_dataset import CocoDataset

OBJ_DETECTION_DATUM_T = tuple[InputType, TargetType, DatumMetadataType]

LOG = logging.getLogger(__name__)


@dataclass
class MAITEDetectionTarget:
    """Dataclass for the datum-level MAITE output detection format."""

    boxes: np.ndarray[Any, Any]
    labels: np.ndarray[Any, Any]
    scores: np.ndarray[Any, Any]


class COCOMetadata(DatumMetadata):
    """TypedDict for COCO-detection datum-level metdata."""

    ann_ids: ReadOnly[Sequence[int]]
    image_info: ReadOnly[dict[str, Any]]


class COCOMAITEObjectDetectionDataset(Dataset):
    """Dataset class to convert a COCO dataset to a dataset compliant with MAITE's Object Detection protocol.

    Parameters
    ----------
    metadata: DatasetMetadata
            Metadata of this dataset.
    """

    def __init__(  # noqa: C901
        self,
        *,
        kwcoco_dataset: CocoDataset,
        image_metadata: Sequence[DatumMetadataType],
        skip_no_anns: bool = False,
        dataset_id: str | None = None,
    ) -> None:
        """Initialize MAITE-compliant dataset from a COCO dataset.

        Args:
            kwcoco_dataset (kwcoco.CocoDataset): The COCO dataset object.
            image_metadata (Sequence[DatumMetadataType]): Metadata for each image.
            skip_no_anns (bool): Whether to skip images without annotations. Defaults to False.
            dataset_id (str): Dataset ID, defaults to filepath.

        Raises:
            ImportError: If required dependencies are not installed.
            ValueError: If metadata is missing for any image in the dataset.
        """
        if not is_usable:
            raise ImportError(
                "COCOMAITEObjectDetectionDataset requires additional dependencies, \
                please install 'xaitk-saliency[tools]'",
            )

        self._kwcoco_dataset = kwcoco_dataset

        self._image_ids = []
        self._annotations = {}

        for _, img_id in enumerate(kwcoco_dataset.imgs.keys()):
            bboxes = np.empty((0, 4))
            labels = []
            scores = []

            if img_id in kwcoco_dataset.gid_to_aids and len(kwcoco_dataset.gid_to_aids[img_id]) > 0:
                det_ids = kwcoco_dataset.gid_to_aids[img_id]
                for det_id in det_ids:
                    ann = kwcoco_dataset.anns[det_id]

                    labels.append(ann["category_id"])

                    if "score" in ann:
                        scores.append(ann["score"])
                    elif "prob" in ann:
                        scores.append(max(ann["prob"]))
                    else:
                        scores.append(1.0)

                    x, y, w, h = ann["bbox"]
                    bbox = [x, y, x + w, y + h]
                    bboxes = np.vstack((bboxes, bbox))
            elif skip_no_anns:
                continue

            img_file_path = Path(kwcoco_dataset.get_image_fpath(img_id))
            if not img_file_path.exists():
                continue
            self._image_ids.append(img_id)
            self._annotations[img_id] = MAITEDetectionTarget(
                boxes=bboxes,
                labels=np.asarray(labels),
                scores=np.asarray(scores),
            )

        self._image_metadata = {
            image_md["id"]: copy.deepcopy(image_md)
            for image_md in image_metadata
            if "id" in image_md and image_md["id"] in self._image_ids
        }
        if len(self._image_metadata) != len(self._image_ids):
            raise ValueError("Image metadata length mismatch, metadata needed for every image.")

        self.metadata: DatasetMetadata = DatasetMetadata(
            id=dataset_id if dataset_id else str(kwcoco_dataset.fpath),
            index2label={c["id"]: c["name"] for c in kwcoco_dataset.cats.values()},
        )

    def __len__(self) -> int:
        """Returns the number of images in the dataset."""
        return len(self._image_ids)

    def __getitem__(self, index: int) -> tuple[InputType, TargetType, COCOMetadata]:
        """Returns the dataset object at the given index."""
        image_id = self._image_ids[index]
        img_file_path = Path(self._kwcoco_dataset.get_image_fpath(image_id))
        image = Image.open(img_file_path)
        width, height = image.size

        gid_to_aids = self._kwcoco_dataset.gid_to_aids

        image_md: COCOMetadata = {
            "id": image_id,
            "ann_ids": (list(gid_to_aids[image_id]) if image_id in gid_to_aids else []),
            "image_info": {"width": width, "height": height, "file_name": img_file_path},
        }

        # Forward input metadata, checking for clobbering
        forwarded_keys = ["id"]
        for key, value in self._image_metadata[image_id].items():
            if key in image_md and key not in forwarded_keys:
                raise KeyError(f"'{key}' already present in metadata, erroring out to prevent overwrite")
            image_md[key] = value

        image_array = np.asarray(image)
        if image.mode == "L":
            image_array = np.expand_dims(image_array, axis=2)

        return (
            np.asarray(np.transpose(image_array, axes=(2, 0, 1))),
            self._annotations[image_id],
            image_md,
        )


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

    def __len__(self) -> int:
        """Returns the number of images in the dataset."""
        return len(self.imgs)

    def __getitem__(self, index: int) -> OBJ_DETECTION_DATUM_T:
        """Returns the dataset object at the given index."""
        return self.imgs[index], self.dets[index], self.datum_metadata[index]
