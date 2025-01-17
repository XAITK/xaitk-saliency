"""Implementation of OcclusionScoring scorer"""

import numpy as np
from sklearn.preprocessing import maxabs_scale

from xaitk_saliency import GenerateClassifierConfidenceSaliency
from xaitk_saliency.utils.masking import weight_regions_by_scalar


class OcclusionScoring(GenerateClassifierConfidenceSaliency):
    """
    This saliency implementation transforms black-box
    image classification scores into saliency heatmaps. This should
    require a sequence of per-class confidences predicted on the
    reference image, a number of per-class confidences as predicted
    on perturbed images, as well as the masks of the reference image
    perturbations (as would be output from a `PerturbImage` implementation).

    The perturbation masks used by the following implementation are
    expected to be of type integer. Masks containing values of type
    float are rounded to the nearest value and binarized
    with value 1 replacing values greater than or equal to half of
    the maximum value in mask after rounding while 0 replaces the rest.
    """

    def generate(
        self,
        reference: np.ndarray,
        perturbed: np.ndarray,
        perturbed_masks: np.ndarray,
    ) -> np.ndarray:
        """
        Generate saliency maps

        :param reference: np.ndarray
            Reference confidence lengths from the reference image
        :param perturbed: np.ndarray
            Perturbed confidence lengths from the reference image
        :param perturbed_masks: np.ndarray
            Perturbation masks `numpy.ndarray` over the reference image.

        :return: np.ndarray
            Generated visual saliency heatmap.
        """
        if len(reference) != len(perturbed[0]):
            raise ValueError("Number of classes in original image and perturbed image do not match.")

        if len(perturbed) != len(perturbed_masks):
            raise ValueError("Number of perturbation masks and respective confidence lengths do not match.")

        # Iterating through each class confidence and compare it with
        # its perturbed twin
        diff = reference - perturbed

        # Weighting perturbed regions with respective difference in confidence
        sal = weight_regions_by_scalar(diff, perturbed_masks)

        # Normalize final saliency map
        sal = maxabs_scale(sal.reshape(sal.shape[0], -1), axis=1).reshape(sal.shape)

        # Ensure saliency map in range [-1, 1]
        return np.clip(sal, -1, 1)

    def get_config(self) -> dict:
        """
        Get the configuration dictionary of the OcclusionScoring instance.

        Returns:
            dict[str, Any]: Configuration dictionary.
        """
        return {}
