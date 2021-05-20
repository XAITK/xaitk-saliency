from xaitk_saliency import ImageClassifierSaliencyMapGenerator
from xaitk_saliency.utils.masking import weight_regions_by_scalar

import numpy as np


class OcclusionScoring (ImageClassifierSaliencyMapGenerator):
    """
    This saliency implementation transforms black-box
    image classification scores into saliency heatmaps. This should
    require a sequence of per-class confidences predicted on the
    reference image, a number of per-class confidences as predicted
    on perturbed images, as well as the masks of the reference image
    perturbations (as would be output from a
    `PerturbImage` implementation.
    """

    def generate(
            self,
            image_conf: np.ndarray,
            perturbed_conf: np.ndarray,
            perturbed_masks: np.ndarray
    ) -> np.ndarray:

        if not (perturbed_masks[0][0][0] in [0, 1]):
            raise ValueError("Image perturbation mask must be of",
                             "type integer and binary valued.")

        try:
            assert len(image_conf) == len(perturbed_conf[0])
        except AssertionError:
            raise ValueError("Number of classses in original image and",
                             " perturbed image do not match.")

        try:
            assert len(perturbed_conf) == len(perturbed_masks)
        except AssertionError:
            raise ValueError("Number of perturbation masks and respective",
                             "confidence lengths do not match.")

        # Iterating through each class confidence and compare it with
        # its perturbed twin
        diff = image_conf - perturbed_conf

        # Weighting perturbed regions with respective difference in confidence
        sal = weight_regions_by_scalar(diff, perturbed_masks)

        return sal

    def get_config(self) -> dict:
        return {}
