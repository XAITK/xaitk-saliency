from typing import Dict, Any
from xaitk_saliency import GenerateClassifierConfidenceSaliency
from xaitk_saliency.utils.masking import weight_regions_by_scalar

import numpy as np
from sklearn.preprocessing import maxabs_scale


class RISEScoring (GenerateClassifierConfidenceSaliency):
    """
    Saliency map generation based on the original RISE implementation.
    This version utilizes only the input perturbed image confidence predictions
    and does not utilize reference image confidences.
    This implementation also takes influence from debiased RISE and may take an
    optional debias probability, `p1` (0 by default).
    In the original paper this is paired with the same probability used in RISE
    perturbation mask generation (see the `p1` parameter in
    :class:`xaitk_saliency.impls.perturb_image.rise.RISEGrid`).

    Based on Hatakeyama et. al:
    https://openaccess.thecvf.com/content/ACCV2020/papers/Hatakeyama_Visualizing_Color-wise_Saliency_of_Black-Box_Image_Classification_Models_ACCV_2020_paper.pdf
    """

    def __init__(
        self,
        p1: float = 0.0,
    ):
        """
        Generate RISE-based saliency maps with optional p1 de-biasing.

        :param p1: De-biasing parameter based on the masking probability.
            This should be a float value in the [0, 1] range.

        :raises ValueError: Input `p1` was not in the [0,1] range.
        """
        if p1 < 0 or p1 > 1:
            raise ValueError(
                f"Input p1 value of {p1} is not within the expected [0,1] "
                f"range."
            )
        self.p1 = p1

    def generate(
        self,
        image_conf: np.ndarray,
        perturbed_conf: np.ndarray,
        perturbed_masks: np.ndarray,
    ) -> np.ndarray:
        if len(perturbed_conf) != len(perturbed_masks):
            raise ValueError("Number of perturbation masks and respective "
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
