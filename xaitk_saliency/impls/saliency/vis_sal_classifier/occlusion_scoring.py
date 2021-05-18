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

        # Iterating through each class confidence and compare it with
        # its perturbed twin
        diff = image_conf - perturbed_conf

        # Weighting perturbed regions with respective difference in confidence
        sal = weight_regions_by_scalar(diff, perturbed_masks)

        return sal

    def get_config(self) -> dict:
        return {}
