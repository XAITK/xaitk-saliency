from typing import Optional, Dict, Any
from xaitk_saliency import ImageClassifierSaliencyMapGenerator
from xaitk_saliency.utils.masking import weight_regions_by_scalar

import numpy as np
from sklearn.preprocessing import maxabs_scale


class RISEScoring (ImageClassifierSaliencyMapGenerator):
    """
    This saliency implementation transforms black-box
    image classification scores into saliency heatmaps. This should
    require a sequence of per-class confidences predicted on the
    reference image, a number of per-class confidences as predicted
    on perturbed images, as well as the masks of the reference image
    perturbations (as would be output from a
    `PerturbImage` implementation.

    The perturbation masks used by the following implementation are
    expected to be of type integer. Masks containing values of type
    float are rounded to the nearest value and binarized
    with value 1 replacing values greater than or equal to half of
    the maximum value in mask after rounding while 0 replaces the rest.
    """

    def __init__(
        self,
        p1: float = 0.0,
    ):
        """
        Generate RISE-based saliency maps with optional p1 de-biasing.

        :param p1: De-biasing parameter based on the masking probability.
        """
        self.p1 = p1

    def generate(
        self,
        image_conf: np.ndarray,
        perturbed_conf: np.ndarray,
        perturbed_masks: np.ndarray,
    ) -> np.ndarray:

        if len(image_conf) != len(perturbed_conf[0]):
            raise ValueError("Number of classes in original image and ",
                             "perturbed image do not match.")

        if len(perturbed_conf) != len(perturbed_masks):
            raise ValueError("Number of perturbation masks and respective ",
                             "confidence lengths do not match.")

        # The RISE method does not use the difference of confidences, but just
        # the perturbed image confidences. The reference confidences are not
        # used here.

        # Weighting perturbed regions with respective difference in confidence
        sal = weight_regions_by_scalar(perturbed_conf,
                                       perturbed_masks - self.p1,
                                       inv_masks=False,
                                       normalize=False)

        # Normalize final saliency map
        sal = maxabs_scale(
            sal.reshape(sal.shape[0], -1),
            axis=1
        ).reshape(sal.shape)

        # Ensure saliency map in range [-1, 1]
        sal = np.clip(sal, -1, 1)

        return sal

    def get_config(self) -> Dict[str, Any]:
        return {
            "p1": self.p1,
        }
