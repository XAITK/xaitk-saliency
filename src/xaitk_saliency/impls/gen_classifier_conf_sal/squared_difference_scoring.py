"""Implementation of SimilarityScoring scorer"""

import numpy as np
from typing_extensions import override

from xaitk_saliency import GenerateClassifierConfidenceSaliency
from xaitk_saliency.utils.masking import weight_regions_by_scalar


class SquaredDifferenceScoring(GenerateClassifierConfidenceSaliency):
    """
    This saliency implementation transforms black-box confidence predictions
    from a classification-style network into saliency heatmaps. This should
    require a sequence of classification scores predicted on the reference
    image, a number of classification scores predicted on perturbed images, as
    well as the masks of the reference image perturbations (as would be output
    from a `PerturbImage` implementation).

    This implementation uses the squared difference of the reference scores and
    the perturbed scores to compute the saliency maps. This gives an indication
    of general saliency without distinguishing between positive and negative.
    The resulting maps are normalized between the range [0,1].

    Based on Greydanus et. al:
    https://arxiv.org/abs/1711.00138
    """

    @override
    def generate(
        self,
        reference: np.ndarray,
        perturbed: np.ndarray,
        perturbed_masks: np.ndarray,
    ) -> np.ndarray:
        """
        Generate saliency heatmaps from black-box confidence predictions

        :param reference: np.ndarray
            Reference predictions from the reference image
        :param perturbed: np.ndarray
            Perturbed predictions from the reference image
        :param perturbed_masks: np.ndarray
            Perturbation masks `numpy.ndarray` over the reference image.

        :return: np.ndarray
            Generated visual saliency heatmap.
        """
        if len(reference) != len(perturbed[0]):
            raise ValueError("Number of classes in original image and perturbed image do not match.")

        if len(perturbed) != len(perturbed_masks):
            raise ValueError("Number of perturbation masks and respective confidence lengths do not match.")

        # Based on equations 2 and 4 from Greydanus et al., '18
        diff = 0.5 * ((reference - perturbed) ** 2).sum(axis=1, keepdims=True)
        sal = weight_regions_by_scalar(diff, perturbed_masks)[0]

        # Normalize saliency map to [0,1]
        sal -= sal.min()
        return sal / sal.max()

    def get_config(self) -> dict:
        """
        Get the configuration dictionary of the SquaredDifferenceScoring instance.

        Returns:
            dict[str, Any]: Configuration dictionary.
        """
        return {}
