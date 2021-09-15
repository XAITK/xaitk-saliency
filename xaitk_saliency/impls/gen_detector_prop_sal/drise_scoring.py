from xaitk_saliency import GenerateDetectorProposalSaliency
from xaitk_saliency.utils.masking import weight_regions_by_scalar

import numpy as np
from sklearn.preprocessing import maxabs_scale
from scipy.spatial.distance import cdist


class DRISEScoring (GenerateDetectorProposalSaliency):
    """
    This D-RISE implementation transforms black-box object detector predictions
    into visual saliency heatmaps. Specifically, we make use of perturbed
    detections generated using the `RISEGrid` image perturbation class and
    a similarity metric that captures both the localization and categorization
    aspects of object detection.

    Object detection representations used here would need to encapsulate
    localization information (i.e. bounding box regions), class scores, and
    objectness scores (if applicable to the detector, such as YOLOv3).
    Object detections are converted into (4+1+nClasses) vectors (4 indices for
    bounding box locations, 1 index for objectness, and nClasses indices for
    different object classes).

    Based on Petsiuk et al:
    https://arxiv.org/abs/2006.03204

    :param proximity_metric: String label for the distance metric to use.
      See scipy.spatial.distance.cdist() and it's *metric* parameter for more
      details and possible values.
    """

    def __init__(
        self,
        proximity_metric: str = 'cosine'
    ):

        try:
            # Attempting to use chosen comparison metric
            cdist([[1], [1]], [[1], [1]], proximity_metric)
            self.proximity_metric: str = proximity_metric
        except ValueError:
            raise ValueError("Chosen comparison metric not supported or "
                             "may not be available in scipy")

    def iou(self, box_a: np.ndarray, box_b: np.ndarray) -> np.ndarray:
        """
        | Compute the intersection over union (IoU) of two sets of boxes.
        | E.g.:
        |    A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)

        :param box_a: (np.array) bounding boxes, Shape: [A,4]
        :param box_b: (np.array) bounding boxes, Shape: [B,4]
        :return: iou(np.array), Shape: [A,B].
        """
        # check input box shape and dimensions
        assert box_a.ndim == 2
        assert box_b.ndim == 2
        assert box_a.shape[1] == 4
        assert box_b.shape[1] == 4

        x11, y11, x12, y12 = np.split(box_a, 4, axis=1)
        x21, y21, x22, y22 = np.split(box_b, 4, axis=1)

        # determine the (x, y)-coordinates of the intersection rectangle
        xa = np.maximum(x11, np.transpose(x21))
        ya = np.maximum(y11, np.transpose(y21))
        xb = np.minimum(x12, np.transpose(x22))
        yb = np.minimum(y12, np.transpose(y22))

        # compute the area of intersection rectangle
        inter_area = np.maximum((xb - xa + 1), 0) * \
            np.maximum((yb - ya + 1), 0)

        # compute the area of both the prediction and ground-truth rectangles
        box_a_area = (x12 - x11 + 1) * (y12 - y11 + 1)
        box_b_area = (x22 - x21 + 1) * (y22 - y21 + 1)

        # add a small eps to prevent divide by zero errors
        iou = inter_area / \
            (box_a_area + np.transpose(box_b_area) -
             inter_area + np.finfo(float).eps)

        return iou

    def generate(
        self,
        ref_dets: np.ndarray,
        perturbed_dets: np.ndarray,
        perturbed_masks: np.ndarray,
    ) -> np.ndarray:

        if len(perturbed_dets) != len(perturbed_masks):
            raise ValueError("Number of perturbation masks and respective "
                             "detections vector do not match.")

        if ref_dets.shape[1] != perturbed_dets.shape[2]:
            raise ValueError("Dimensions of reference detections and "
                             "perturbed detections do not match. Both "
                             "should be of dimension (n_classes + 4 + 1).")

        n_masks = len(perturbed_masks)
        n_props = perturbed_dets.shape[1]
        n_dets = len(ref_dets)

        # Compute IoU of bounding boxes
        s1 = self.iou(perturbed_dets[:, :, :4].reshape(-1, 4),
                      ref_dets[:, :4]).reshape(n_masks, n_props, n_dets)

        # Compute similarity of class probabilities
        s2 = cdist(perturbed_dets[:, :, 5:].reshape(n_masks * n_props, -1),
                   ref_dets[:, 5:],
                   metric=self.proximity_metric).reshape(n_masks, n_props, n_dets)

        # Use objectness score if available
        s3 = perturbed_dets[:, :, 4:5]

        # Compute overall similarity s
        # Shape: n_masks x n_props x n_dets
        s = s1 * s2 * s3

        # Take max similarity over all proposals
        # Shape: n_masks x n_dets
        s = s.max(axis=1)

        # Weighting perturbed regions by similarity
        sal = weight_regions_by_scalar(s, perturbed_masks)

        # Normalize final saliency map
        sal = maxabs_scale(
            sal.reshape(sal.shape[0], -1),
            axis=1
        ).reshape(sal.shape)

        # Ensure saliency map in range [-1, 1]
        sal = np.clip(sal, -1, 1)

        return sal

    def get_config(self) -> dict:
        return {
            "proximity_metric": self.proximity_metric,
        }
