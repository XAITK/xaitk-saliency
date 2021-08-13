import numpy as np
from xaitk_saliency import GenerateDetectorProposalSaliency

import torch
from scipy.spatial.distance import cdist
import sklearn.preprocessing


class DetectorRISE (GenerateDetectorProposalSaliency):
    """
    This interface proposes that implementations transform black-box image
    object detection predictions into visual saliency heatmaps.
    This should require externally-generated object detection predictions over
    some image, along with predictions for perturbed images and the permutation
    masks for those images as would be output from a
    :class:`xaitk_saliency.interfaces.perturb_image.PerturbImage`
    implementation.

    Object detection representations used here would need to encapsulate
    localization information (i.e. bounding box regions), class scores, and
    objectness scores (if applicable to the detector, such as YOLOv3).
    Object detections are converted into (4+1+nClasses) vectors (4 indices for
    bounding box locations, 1 index for objectness, and nClasses indices for
    different object classes).
    """
    def __init__(
        self,
        proximity_metric: str = 'cosine'
    ):

        try:
            # Attempting to use chosen comparision metric
            cdist([[1], [1]], [[1], [1]], proximity_metric)
            self.proximity_metric: str = proximity_metric
        except ValueError:
            raise ValueError("Chosen comparision metric not supported or",
                             "may not be available in scipy")

    def intersect(self, box_a: torch.FloatTensor, box_b: torch.FloatTensor) -> torch.FloatTensor:
        """ We resize both tensors to [A,B,2] without new malloc:
        [A,2] -> [A,1,2] -> [A,B,2]
        [B,2] -> [1,B,2] -> [A,B,2]
        Then we compute the area of intersect between box_a and box_b.
        Args:
          box_a: (tensor) bounding boxes, Shape: [A,4].
          box_b: (tensor) bounding boxes, Shape: [B,4].
        Return:
          (tensor) intersection area, Shape: [A,B].
        """
        A = box_a.size(0)
        B = box_b.size(0)
        min_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                           box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
        max_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                           box_b[:, :2].unsqueeze(0).expand(A, B, 2))
        inter = torch.clamp((min_xy - max_xy + 1), min=0)
        return inter[:, :, 0] * inter[:, :, 1]

    def jaccard(self, box_a: torch.FloatTensor, box_b: torch.FloatTensor) -> torch.FloatTensor:
        """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
        is simply the intersection over union of two boxes.  Here we operate on
        ground truth boxes and default boxes.
        E.g.:
            A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
        Args:
            box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
            box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
        Return:
            jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
        """
        inter = self.intersect(box_a, box_b)
        area_a = ((box_a[:, 2]-box_a[:, 0]) *
                  (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)
        area_b = ((box_b[:, 2]-box_b[:, 0]) *
                  (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)
        union = area_a + area_b - inter

        return inter / union

    def generate(
        self,
        ref_dets: np.ndarray,
        perturbed_dets: np.ndarray,
        perturb_masks: np.ndarray,
    ) -> np.ndarray:

        if len(perturb_masks) != len(perturbed_dets):
            raise ValueError("Number of perturbation masks and respective",
                             "detections vector do not match.")

        self.n_targets = len(ref_dets)
        n_targets = len(ref_dets)
        self.N = len(perturb_masks)
        self.input_size = perturb_masks[0].shape
        self.masks = torch.from_numpy(perturb_masks)
        return_array = np.empty((n_targets, *self.input_size))

        for i, dete_ in enumerate(ref_dets):
            this_dete = dete_.reshape(-1, len(dete_))[:, :4]
            p_dete = perturbed_dets[:, i, :]
            s1 = self.jaccard(torch.from_numpy(p_dete[:, :4]), torch.from_numpy(this_dete))
            s2 = cdist(dete_[5:].reshape(1, -1),
                       perturbed_dets[:, i, :][:, 5:],
                       metric=self.proximity_metric)
            s2 = torch.tensor(sklearn.preprocessing.minmax_scale(s2, feature_range=(0, 1), axis=1, copy=True))
            try:
                s3 = torch.tensor(perturbed_dets[:, i, 6])
            except IndexError:
                s3 = torch.ones((self.N))
            cur_weights = s1 * s2 * s3
            cur_weights = cur_weights.max(dim=1)[0].t()
            # TODO: Replace with masking.py utils fnc()
            sals = cur_weights.reshape(
                1, -1).mm(
                self.masks.view(self.N, -1).double()
                ).view(1, *self.input_size) / self.masks.sum(0)
            return_array[i, :self.input_size[0], :self.input_size[1]] = sals[0].cpu().numpy()
        return return_array

    def get_config(self) -> dict:
        return {
            "proximity_metric": self.proximity_metric,
        }
