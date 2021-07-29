from xaitk_saliency import GenerateDescriptorSimilaritySaliency
from xaitk_saliency.utils.masking import weight_regions_by_scalar

import numpy as np
from sklearn.preprocessing import maxabs_scale
from scipy.spatial.distance import cdist


class SimilarityScoring (GenerateDescriptorSimilaritySaliency):
    """
    This saliency implementation transforms proximity in feature
    space into saliency heatmaps. This should
    require a sequence of feature vectors of the query and
    reference image, a number of feature vectors as predicted
    on perturbed images, as well as the masks of the reference image
    perturbations (as would be output from a
    `PerturbImage` implementation.

    The perturbation masks used by the following implementation are
    expected to be of type integer. Masks containing values of type
    float are rounded to the nearest value and binarized
    with value 1 replacing values greater than or equal to half of
    the maximum value in mask after rounding while 0 replaces the rest.

    param proximity_metric: The type of comparison metric used
        to determine proximity in feature space. The type of comparison
        metric supported is restricted by scipy's cdist() function. The
        following metrics are supported in scipy.

        ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’,
        ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘jensenshannon’,
        ‘kulsinski’, ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’,
        ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’,
        ‘sqeuclidean’, ‘wminkowski’, ‘yule’.
    """

    def __init__(
        self,
        proximity_metric: str = 'euclidean'
    ):

        try:
            # Attempting to use chosen comparision metric
            cdist([[1], [1]], [[1], [1]], proximity_metric)
            self.proximity_metric: str = proximity_metric
        except ValueError:
            raise ValueError("Chosen comparision metric not supported or",
                             "may not be available in scipy")

    def generate(
        self,
        ref_descr_1: np.ndarray,
        ref_descr_2: np.ndarray,
        perturbed_descrs: np.ndarray,
        perturbed_masks: np.ndarray,
    ) -> np.ndarray:

        if len(perturbed_descrs) != len(perturbed_masks):
            raise ValueError("Number of perturbation masks and respective",
                             "feature vector do not match.")

        if len(ref_descr_1) != len(ref_descr_2):
            raise ValueError("Length of feature vector between",
                             "two images do not match.")

        # Computing original proximity between image1 and image2 feature vectors.
        original_proximity = cdist(
            ref_descr_1.reshape(1, -1),
            ref_descr_2.reshape(1, -1),
            metric=self.proximity_metric
        )

        # Computing proximity between original image1 and perturbed image2 feature vectors.
        perturbed_proximity = cdist(
            ref_descr_1.reshape(1, -1),
            perturbed_descrs,
            metric=self.proximity_metric
        )[0]

        # Iterating through each distance and compare it with
        # its perturbed twin
        diff = perturbed_proximity - original_proximity

        diff = np.transpose(np.clip(diff, 0, None))

        # Weighting perturbed regions with respective difference in confidence
        sal = weight_regions_by_scalar(diff, perturbed_masks)

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
