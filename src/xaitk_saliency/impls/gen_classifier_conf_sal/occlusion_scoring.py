"""Implementation of OcclusionScoring scorer."""

from typing import Any

import numpy as np
from sklearn.preprocessing import maxabs_scale

from xaitk_saliency import GenerateClassifierConfidenceSaliency
from xaitk_saliency.utils.masking import weight_regions_by_scalar


class OcclusionScoring(GenerateClassifierConfidenceSaliency):
    """This saliency implementation transforms black-box image classification scores into saliency heatmaps.

    This should require a sequence of per-class confidences predicted on the
    reference image, a number of per-class confidences as predicted
    on perturbed images, as well as the masks of the reference image
    perturbations (as would be output from a `PerturbImage` implementation).

    The perturbation masks used by the following implementation are
    expected to be of type integer. Masks containing values of type
    float are rounded to the nearest value and binarized
    with value 1 replacing values greater than or equal to half of
    the maximum value in mask after rounding while 0 replaces the rest.

    Example:
        >>> n_classes, n_masks, height, width = 2, 3, 4, 5
        >>> reference = np.array([0.6, 0.4])
        >>> perturbed = np.array([[0.2, 0.8], [0.9, 0.1], [0.5, 0.5]])
        >>> perturbed_masks = np.broadcast_to(np.eye(height, width), (n_masks, height, width))
        >>> scorer = OcclusionScoring()
        >>> sal = scorer(reference=reference, perturbed=perturbed, perturbed_masks=perturbed_masks)
        >>> sal.shape == (n_classes, height, width)
        True
    """

    def generate(
        self,
        *,
        reference: np.ndarray[Any, Any],
        perturbed: np.ndarray[Any, Any],
        perturbed_masks: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """Generate saliency maps.

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
        sal = weight_regions_by_scalar(scalar_vec=diff, masks=perturbed_masks)

        # Normalize final saliency map
        sal = maxabs_scale(sal.reshape(sal.shape[0], -1), axis=1).reshape(sal.shape)

        # Ensure saliency map in range [-1, 1]
        return np.clip(sal, -1, 1)

    def get_config(self) -> dict[str, Any]:
        """Get the configuration dictionary of the OcclusionScoring instance.

        Returns:
            dict[str, Any]: Configuration dictionary.
        """
        return {}
