from xaitk_saliency import GenerateDescriptorSimilaritySaliency
from xaitk_saliency.utils.masking import weight_regions_by_scalar

import numpy as np
from sklearn.preprocessing import maxabs_scale
from scipy.spatial.distance import cdist


class SimilarityScoring (GenerateDescriptorSimilaritySaliency):
    """
    This saliency implementation transforms proximity in feature space into
    saliency heatmaps.
    This should require feature vectors for the reference image, for each query
    image, and for perturbed versions of the reference image, as well as the
    masks of the reference image perturbations (as would be output from a
    `PerturbImage` implementation).

    The resulting saliency maps are relative to the reference image.
    As such, each map denotes regions in the reference image that make it more
    or less similar to the corresponding query image.

    :param proximity_metric: The type of comparison metric used
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
            # Attempting to use chosen comparison metric
            cdist([[1], [1]], [[1], [1]], proximity_metric)
            self.proximity_metric: str = proximity_metric
        except ValueError:
            raise ValueError("Chosen comparison metric not supported or "
                             "may not be available in scipy")

    def generate(
        self,
        ref_descr: np.ndarray,
        query_descrs: np.ndarray,
        perturbed_descrs: np.ndarray,
        perturbed_masks: np.ndarray,
    ) -> np.ndarray:
        if len(perturbed_descrs) != len(perturbed_masks):
            raise ValueError("Number of perturbation masks and respective "
                             "feature vector do not match.")

        if len(ref_descr) != query_descrs.shape[1]:
            raise ValueError("Size of feature vectors between reference and "
                             "query images do not match.")

        # Computing original proximity between reference image feature vector
        # and each query image feature vector.
        original_proximity = cdist(
            ref_descr.reshape(1, -1),
            query_descrs,
            metric=self.proximity_metric
        )

        # Computing proximity between query feature vectors and perturbed
        # reference image feature vectors.
        perturbed_proximity = cdist(
            query_descrs,
            perturbed_descrs,
            metric=self.proximity_metric
        )

        # Iterating through each distance and compare it with
        # its perturbed twin
        diff = perturbed_proximity - np.transpose(original_proximity)

        diff = np.transpose(np.clip(diff, 0, None))

        # Weighting perturbed regions with respective difference in confidence
        sal = weight_regions_by_scalar(diff, perturbed_masks)

        # Normalize final saliency maps
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
