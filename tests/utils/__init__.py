from xaitk_saliency import PerturbImage, GenerateDetectorProposalSaliency
from smqtk_detection import DetectImageObjects
from smqtk_detection.utils.bbox import AxisAlignedBoundingBox
from typing import Iterable, Dict, Hashable, Tuple
import numpy as np


class TestDetector(DetectImageObjects):
    """
    Dummy detector that returns constant detections.
    """

    def detect_objects(
        self,
        img_iter: Iterable[np.ndarray]
    ) -> Iterable[Iterable[Tuple[AxisAlignedBoundingBox, Dict[Hashable, float]]]]:
        for i, img in enumerate(img_iter):
            yield [(AxisAlignedBoundingBox((0, 0), (100, 100)), {0: 1.0, "cat1": 0.0, 2: 0.0})] * i

    def get_config(self) -> dict: ...


class MismatchedDetector(DetectImageObjects):
    """
    Dummy detector that has a class label not in the test detections.
    """

    def detect_objects(
        self,
        img_iter: Iterable[np.ndarray]
    ) -> Iterable[Iterable[Tuple[AxisAlignedBoundingBox, Dict[Hashable, float]]]]:
        for i, img in enumerate(img_iter):
            yield [(AxisAlignedBoundingBox((0, 0), (100, 100)), {3: 1.0})] * i

    def get_config(self) -> dict: ...


class MismatchedNameDetector(DetectImageObjects):
    """
    Dummy detector that has a class label not in the test detections. This
    label is a string rather than an id.
    """

    def detect_objects(
        self,
        img_iter: Iterable[np.ndarray]
    ) -> Iterable[Iterable[Tuple[AxisAlignedBoundingBox, Dict[Hashable, float]]]]:
        for i, img in enumerate(img_iter):
            yield [(AxisAlignedBoundingBox((0, 0), (100, 100)), {"cat3": 1.0})] * i

    def get_config(self) -> dict: ...


class TestPerturber(PerturbImage):
    """
    Dummy perturber that returns four masks, each with different
    corners of the image occluded.
    """

    def perturb(
        self,
        ref_image: np.ndarray
    ) -> np.ndarray:
        img_h = ref_image.shape[0]
        img_w = ref_image.shape[1]

        masks = np.ones((4, img_h, img_w))
        masks[0, 0:img_h//2, 0:img_w//2] = 0
        masks[1, 0:img_h//2, img_w//2:img_w] = 0
        masks[2, img_h//2:img_h, 0:img_w//2] = 0
        masks[3, img_h//2:img_h, img_w//2:img_w] = 0

        return masks

    def get_config(self) -> dict: ...


class TestSalGenerator(GenerateDetectorProposalSaliency):
    """
    Dummy detection saliency generator. Combines perturbed detections
    and reference detection in an arbitrary way to produce maps.
    """

    def generate(
        self,
        ref_dets: np.ndarray,
        perturb_dets: np.ndarray,
        perturb_masks: np.ndarray,
    ) -> np.ndarray:

        m = [
            pert_det.sum() * mask
            for pert_det, mask in zip(perturb_dets, perturb_masks)
        ]

        sal_maps = np.asarray([
            ref_det.sum() * m[i]
            for i, ref_det in enumerate(ref_dets)
        ])

        sal_maps = sal_maps / sal_maps.max()

        return sal_maps

    def get_config(self) -> dict: ...
