"""This module provides the `MAITEImageClassifier` class.

It is an adapter for integrating MAITE-based image classifiers with the SMQTK `ClassifyImage` interface. It enables
the use of MAITE protocol-based classifiers in pipelines that require the SMQTK interface.

Classes:
    MAITEImageClassifier: Adapts a MAITE image classification model for compatibility with
                          SMQTK's `ClassifyImage` interface.

Dependencies:
    - maite.protocols.image_classification: For the MAITE image classification protocol.
    - numpy: For numerical operations.
    - smqtk_classifier: For SMQTK classifier interfaces and classification utilities.
"""

from collections.abc import Hashable, Iterator, Sequence

import maite.protocols.image_classification as ic
import numpy as np
from smqtk_classifier.interfaces.classification_element import CLASSIFICATION_DICT_T
from smqtk_classifier.interfaces.classify_image import IMAGE_ITER_T, ClassifyImage
from typing_extensions import override


class MAITEImageClassifier(ClassifyImage):
    """Adapter for the MAITE image classification protocol, implementing the SMQTK `ClassifyImage` interface.

    This adapter allows a MAITE protocol-based classifier to be used in SMQTK pipelines by transforming
    classification outputs into the expected format.

    Attributes:
        _classifier (ic.Model): The MAITE protocol-based image classifier instance.
        _ids (dict[int, Hashable]): A dictionary mapping label IDs to human-readable names.
        _img_batch_size (int): The number of images to process in a single batch.

    Methods:
        get_labels: Retrieve the human-readable class labels.
        classify_images: Perform classification on a batch of input images.
        get_config: Raise a `NotImplementedError` for serialization (not yet supported).
    """

    def __init__(
        self,
        *,
        classifier: ic.Model,
        ids: Sequence[int],
        img_batch_size: int = 1,
    ) -> None:
        """Initialize the MAITEImageClassifier with a MAITE protocol-based classifier.

        Args:
            classifier (ic.Model): The MAITE protocol-based image classification model.
            ids (dict[int, Hashable]): A dictionary mapping label IDs to human-readable names.
            img_batch_size (int, optional): The number of images to process in a single batch. Defaults to 1.
        """
        self._classifier = classifier
        self._ids = sorted(ids)
        self._img_batch_size = img_batch_size

    @override
    def get_labels(self) -> Sequence[Hashable]:
        return self._ids

    @override
    def classify_images(self, img_iter: IMAGE_ITER_T) -> Iterator[CLASSIFICATION_DICT_T]:  # noqa: C901
        all_out = list()
        batch = list()

        # Convert from channels last to channels first
        # Channels first is specified in protocols v0.5.0
        def _to_channels_first(img: np.ndarray) -> np.ndarray:
            if img.ndim < 3:
                return img
            return np.moveaxis(img, -1, 0)

        # Transform outputs of a single batch
        def _generate_outputs(batch: Sequence[np.ndarray]) -> list[dict[int, float]]:
            predictions = np.asarray(self._classifier(batch))

            # Note: We make an assumption that MAITE outputs will be ordered
            return [{self._ids[idx]: value for idx, value in enumerate(pred)} for pred in predictions]

        # Batch model passes
        for img in img_iter:
            batch.append(_to_channels_first(img))

            if len(batch) == self._img_batch_size:
                all_out.extend(_generate_outputs(batch))
                batch = list()

        # Leftover batch
        if len(batch) > 0:
            _generate_outputs(batch)

        return iter(all_out)

    @override
    def get_config(self) -> dict:
        raise NotImplementedError(
            "Constructor arguments are not serializable as is and require further implementation to do so.",
        )
