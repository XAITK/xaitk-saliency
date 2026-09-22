"""Fakes and test doubles shared across the test suite."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Iterator, Sequence
from typing import Any

import numpy as np
from smqtk_classifier.interfaces.classify_image import ClassifyImage
from smqtk_descriptors.interfaces.image_descriptor_generator import ImageDescriptorGenerator
from smqtk_detection.interfaces.detect_image_objects import DetectImageObjects
from smqtk_image_io.bbox import AxisAlignedBoundingBox
from typing_extensions import override


class FakeClassifyImage(ClassifyImage):
    """Fake `ClassifyImage` implementation that returns the same fixed confidences for every image."""

    def __init__(self, confidences: dict[Hashable, float]) -> None:
        self._confidences = confidences

    @override
    def get_labels(self) -> Sequence[Hashable]:
        return list(self._confidences)

    @override
    def classify_images(self, img_iter: Iterable[np.ndarray]) -> Iterator[dict[Hashable, float]]:
        for _ in img_iter:
            yield self._confidences

    @override
    def get_config(self) -> dict[str, Any]:
        return {}


class FakeImageDescriptorGenerator(ImageDescriptorGenerator):
    """Fake `ImageDescriptorGenerator` implementation that returns a fixed zero vector for every image."""

    @override
    def generate_arrays_from_images(self, img_mat_iter: Iterable[np.ndarray | None]) -> Iterator[np.ndarray]:
        for _ in img_mat_iter:
            yield np.zeros(4)

    @override
    def get_config(self) -> dict[str, Any]:
        return {}


class FakeDetectImageObjects(DetectImageObjects):
    """Fake `DetectImageObjects` implementation that returns the same fixed detection for every image."""

    @override
    def detect_objects(
        self,
        img_iter: Iterable[np.ndarray],
    ) -> Iterator[list[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]]:
        for _ in img_iter:
            yield [(AxisAlignedBoundingBox((0, 0), (8, 8)), {"class0": 0.3, "class1": 0.7})]

    @override
    def get_config(self) -> dict[str, Any]:
        return {}
