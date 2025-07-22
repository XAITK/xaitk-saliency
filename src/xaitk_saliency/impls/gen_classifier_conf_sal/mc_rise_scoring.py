"""Implementation of MC-RISE scorer"""

from typing import Any

import numpy as np
from sklearn.preprocessing import maxabs_scale
from typing_extensions import override

from xaitk_saliency import GenerateClassifierConfidenceSaliency
from xaitk_saliency.utils.masking import weight_regions_by_scalar


class MCRISEScoring(GenerateClassifierConfidenceSaliency):
    """
    Saliency map generation based on the MC-RISE implementation.
    This version utilizes only the input perturbed image confidence predictions
    and does not utilize reference image confidences.
    This implementation also takes influence from debiased RISE and may take an
    optional debias probability, `p1` (0 by default).
    In the original paper this is paired with the same probability used in RISE
    perturbation mask generation (see the `p1` parameter in
    :class:`xaitk_saliency.impls.perturb_image.mc_rise.MCRISEGrid`).

    Based on Hatakeyama et. al:
    https://openaccess.thecvf.com/content/ACCV2020/papers/Hatakeyama_Visualizing_Color-wise_Saliency_of_Black-Box_Image_Classification_Models_ACCV_2020_paper.pdf
    """

    def __init__(
        self,
        k: int,
        p1: float = 0.0,
    ) -> None:
        """
        :param k: int
            Number of colors to used during perturbation.
        :param p1: float
            Debias probability, typically paired with the same probability used in mask generation.

        :raises: ValueError
            If p1 not in in [0, 1].
        :raises: ValueError
            If k < 1.
        """
        if p1 < 0 or p1 > 1:
            raise ValueError(f"Input p1 value of {p1} is not within the expected [0,1] range.")
        self.p1 = p1

        if k < 1:
            raise ValueError(f"Input k value of {k} is not within the expected >0 range.")
        self.k = k

    @override
    def generate(
        self,
        reference: np.ndarray,
        perturbed: np.ndarray,
        perturbed_masks: np.ndarray,
    ) -> np.ndarray:
        """
        Warning: this implementation returns a different shape than typically expected by
        this interface. Instead of `[nClasses x H x W]`, saliency maps of shape
        `[kColors x nClasses x H x W]` are generated, one per color per class.

        :param reference: np.ndarray
            Reference image predicted class-confidence vector, as a
            `numpy.ndarray`, for all classes that require saliency map
            generation.
            This should have a shape `[nClasses]`, be float-typed and with
            values in the [0,1] range.
        :param perturbed: np.ndarray
            Perturbed image predicted class confidence matrix.
            Classes represented in this matrix should be congruent to classes
            represented in the `reference` vector.
            This should have a shape `[nMasks x nClasses]`, be float-typed and
            with values in the [0,1] range.
        :param perturbed_masks: np.ndarray
            Perturbation masks `numpy.ndarray` over the reference image.
            This should be parallel in association to the classification
            results input into the `perturbed` parameter.
            This should have a shape `[kColors x nMasks x H x W]`, and values in range
            [0, 1], where a value closer to 1 indicate areas of the image that
            are *unperturbed*.

        :return: np.ndarray
            Generated visual saliency heatmap for each input class as a
            float-type `numpy.ndarray` of shape `[kColors x nClasses x H x W]`.

        :raises: ValueError
            If number of perturbations masks and respective confidence lengths do not match.
        """
        if len(perturbed) != perturbed_masks.shape[1]:
            raise ValueError("Number of perturbation masks and respective confidence lengths do not match.")

        sal_maps = []
        # Compute unmasked regions
        perturbed_masks = 1 - perturbed_masks
        m0 = 1 - perturbed_masks.sum(axis=0)
        for k_masks in perturbed_masks:
            # Debias based on the MC-RISE paper
            sal = weight_regions_by_scalar(
                scalar_vec=perturbed,
                # Factoring out denominator from self.k * k_masks / self.p1 - m0 / (1 - self.p1)
                # to avoid divide by zero. Only acts as a normalization factor
                masks=self.k * k_masks * (1 - self.p1) - m0 * self.p1,
                inv_masks=False,
                normalize=False,
            )

            # Normalize final saliency map
            sal = maxabs_scale(sal.reshape(sal.shape[0], -1), axis=1).reshape(sal.shape)

            # Ensure saliency map in range [-1, 1]
            sal = np.clip(sal, -1, 1)

            sal_maps.append(sal)

        return np.asarray(sal_maps)

    @override
    def get_config(self) -> dict[str, Any]:
        return {
            "p1": self.p1,
            "k": self.k,
        }
